from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class OANRefiner(nn.Module):
    """Occlusion-Aware reconstruction refiner.

    The module is designed for the DE-GAN + OAN reconstruction framework:
    it receives a masked image patch, an occlusion mask, and the frozen DE-GAN
    coarse output patch. It predicts a small residual over the coarse output
    and composites only inside the mask, so unmasked pixels are preserved by
    construction.
    """

    def __init__(self, base_channels: int = 32, residual_scale: float = 0.65):
        super().__init__()
        self.residual_scale = residual_scale
        self.enc1 = _ConvBlock(7, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.enc3 = _ConvBlock(base_channels * 2, base_channels * 4)
        self.bridge = nn.Sequential(
            _ConvBlock(base_channels * 4, base_channels * 4),
            _ResidualBlock(base_channels * 4),
            _ResidualBlock(base_channels * 4),
        )
        self.up3 = self._up(base_channels * 4, base_channels * 4)
        self.dec3 = _ConvBlock(base_channels * 8, base_channels * 2)
        self.up2 = self._up(base_channels * 2, base_channels * 2)
        self.dec2 = _ConvBlock(base_channels * 4, base_channels)
        self.up1 = self._up(base_channels, base_channels)
        self.dec1 = _ConvBlock(base_channels * 2, base_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _up(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        masked_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        degan_patch: torch.Tensor,
        *,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask_patch.shape[1] != 1:
            mask_patch = mask_patch.mean(dim=1, keepdim=True)
        mask_patch = mask_patch.clamp(0.0, 1.0)
        x = torch.cat([masked_patch, degan_patch, mask_patch], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bridge(F.avg_pool2d(e3, 2))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw_refined = (degan_patch + self.delta_head(d1) * self.residual_scale).clamp(-1.0, 1.0)
        final = masked_patch * (1.0 - mask_patch) + raw_refined * mask_patch
        if return_raw:
            return final, raw_refined
        return final


class _UNetCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.enc3 = _ConvBlock(base_channels * 2, base_channels * 4)
        self.bridge = nn.Sequential(
            _ConvBlock(base_channels * 4, base_channels * 4),
            _ResidualBlock(base_channels * 4),
        )
        self.up3 = OANRefiner._up(base_channels * 4, base_channels * 4)
        self.dec3 = _ConvBlock(base_channels * 8, base_channels * 2)
        self.up2 = OANRefiner._up(base_channels * 2, base_channels * 2)
        self.dec2 = _ConvBlock(base_channels * 4, base_channels)
        self.up1 = OANRefiner._up(base_channels, base_channels)
        self.dec1 = _ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bridge(F.avg_pool2d(e3, 2))
        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec1(torch.cat([d1, e1], dim=1)))


class OANGatedFusion(nn.Module):
    """Learned gated DE-GAN + OAN fusion module.

    This matches the checkpoint trained in
    `outputs/degan_oan_fusion_followup_13June/checkpoints/gated_degan_oan`.
    It receives masked patch, occlusion mask, DE-GAN patch, and a context-filled
    patch, then learns a per-pixel gate between the DE-GAN-corrected prior and
    local context. The caller still composites strictly inside the mask.
    """

    def __init__(self, base_channels: int = 32, residual_scale: float = 0.45):
        super().__init__()
        self.residual_scale = residual_scale
        self.gate_smooth_kernel = 0
        self.core = _UNetCore(10, 4, base_channels)

    def forward(
        self,
        masked_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        degan_patch: torch.Tensor,
        context_patch: torch.Tensor,
        *,
        return_raw: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask_patch.shape[1] != 1:
            mask_patch = mask_patch.mean(dim=1, keepdim=True)
        mask_patch = mask_patch.clamp(0.0, 1.0)
        out = self.core(torch.cat([masked_patch, degan_patch, context_patch, mask_patch], dim=1))
        delta = torch.tanh(out[:, :3]) * self.residual_scale
        gate = torch.sigmoid(out[:, 3:4])
        kernel = int(getattr(self, "gate_smooth_kernel", 0) or 0)
        if kernel >= 3:
            if kernel % 2 == 0:
                kernel += 1
            gate = F.avg_pool2d(gate, kernel_size=kernel, stride=1, padding=kernel // 2).clamp(0.0, 1.0)
        degan_corrected = (degan_patch + delta).clamp(-1.0, 1.0)
        raw_refined = (gate * degan_corrected + (1.0 - gate) * context_patch).clamp(-1.0, 1.0)
        final = masked_patch * (1.0 - mask_patch) + raw_refined * mask_patch
        if return_gate:
            return final, raw_refined, gate
        if return_raw:
            return final, raw_refined
        return final
