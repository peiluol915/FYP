import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class GatedConv2d(nn.Module):
    """Gated convolution that learns to ignore masked/corrupted pixels.

    The gating mechanism produces a sigmoid mask that suppresses features
    originating from occluded regions, preventing white-box artifacts from
    leaking through skip connections.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv_feature(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.activation(self.bn(feature * gate))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        channel_att = self.sigmoid(out)
        x = x * channel_att

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        x = x * spatial_att
        return x


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator with spectral normalization.

    Classifies 14x14 patches of the input as real or fake, providing
    spatially-aware adversarial feedback that encourages realistic textures
    in the reconstructed face region.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512, use_cbam=True):
        super(MobileFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)

        self.dw_conv = nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.prelu3 = nn.PReLU(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.prelu4 = nn.PReLU(256)

        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.prelu5 = nn.PReLU(512)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(512)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.dw_conv(x)))
        x = self.prelu3(self.bn3(self.conv2(x)))
        x = self.prelu4(self.bn4(self.conv3(x)))
        x = self.prelu5(self.bn5(self.conv4(x)))
        if self.use_cbam:
            x = self.cbam(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class OAN(nn.Module):
    def __init__(self, backbone_type='resnet50', use_cbam=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.use_cbam = use_cbam
        if backbone_type == 'resnet50':
            self.backbone = resnet50(weights=None)
            if self.use_cbam:
                self.oam = CBAM(2048)
            self.fc = nn.Linear(2048, 512)
        elif backbone_type == 'mobilefacenet':
            self.backbone = MobileFaceNet(embedding_size=512, use_cbam=use_cbam)
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            if self.use_cbam:
                x = self.oam(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return self.fc(x)
        elif self.backbone_type == 'mobilefacenet':
            return self.backbone(x)


class DEGAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Gated first encoder layer suppresses mask pixel features before
        # they can propagate through skip connections to the decoder.
        self.enc1_gate = GatedConv2d(4, 64)
        self.enc1 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.residual_stack = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # CBAM at each skip-merge point lets the decoder attend to valid
        # encoder features and suppress corrupted ones from masked regions.
        self.skip_attn3 = CBAM(512, reduction=8)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.skip_attn2 = CBAM(256, reduction=8)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip_attn1 = CBAM(128, reduction=8)
        self.dec1 = ConvBlock(128, 64)
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            if x.shape[1] == 4:
                network_input = x
            else:
                mask = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
                network_input = torch.cat([x, mask], dim=1)
        else:
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            network_input = torch.cat([x, mask], dim=1)

        enc1 = self.enc1(self.enc1_gate(network_input))
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        bottleneck = self.residual_stack(bottleneck)

        dec3 = self.up3(bottleneck)
        dec3 = self.dec3(self.skip_attn3(torch.cat([dec3, enc3], dim=1)))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(self.skip_attn2(torch.cat([dec2, enc2], dim=1)))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(self.skip_attn1(torch.cat([dec1, enc1], dim=1)))
        return self.output(dec1)


class MTRUNet(nn.Module):
    """Mask-Aware Texture Refinement U-Net.

    This lightweight residual refiner runs after DEGAN. It receives the occluded
    face, the DEGAN output, and a single-channel occlusion mask, predicts a small
    residual texture delta, and blends only the masked region back into the
    original input.
    """

    def __init__(self, base_channels: int = 32, delta_scale: float = 0.75):
        super().__init__()
        self.delta_scale = delta_scale
        self.enc1 = GatedConv2d(7, base_channels)
        self.enc1_refine = ConvBlock(base_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        occluded_face: torch.Tensor,
        degan_output: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        mask = mask.clamp(0.0, 1.0)
        hard_mask = F.max_pool2d((mask > 0.03).to(mask.dtype), kernel_size=7, stride=1, padding=3)
        hard_mask = F.max_pool2d(hard_mask, kernel_size=5, stride=1, padding=2)
        context_face = occluded_face * (1.0 - hard_mask) + degan_output.detach() * hard_mask
        x = torch.cat([context_face, degan_output, hard_mask], dim=1)

        enc1 = self.enc1_refine(self.enc1(x))
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        if dec3.shape[-2:] != enc3.shape[-2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        if dec2.shape[-2:] != enc2.shape[-2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        if dec1.shape[-2:] != enc1.shape[-2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        delta = self.delta_head(dec1) * self.delta_scale
        refined = (degan_output + delta).clamp(-1.0, 1.0)
        final = occluded_face * (1.0 - hard_mask) + refined * hard_mask
        if return_delta:
            return final, delta
        return final
