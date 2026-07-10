param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ProjectRoot "outputs\refiner_clarity_vggface2_$Timestamp.log"
$BackupDir = Join-Path $ProjectRoot "weights\backups\mtr_before_clarity_refiner_$Timestamp"

$env:PYTHONWARNINGS = "ignore::FutureWarning"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Backing up current MTR-UNet refiner weights..."
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$RefinerWeights = @(
  "weights\mtr_unet_best.pth",
  "weights\mtr_unet_final.pth",
  "weights\mtr_unet_disc_final.pth"
)
foreach ($weight in $RefinerWeights) {
  if (Test-Path $weight) {
    Copy-Item $weight (Join-Path $BackupDir (Split-Path $weight -Leaf)) -Force
  }
}

$Manifest = [ordered]@{
  created_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  purpose = "Backup before training a new clarity-focused MTR-UNet refiner against the protected VGGFace2 DE-GAN checkpoint."
  active_degan = "weights\degan_model_best_good_checkpoint.pth"
  backup_dir = $BackupDir
  files = @()
}
foreach ($weight in $RefinerWeights) {
  if (Test-Path $weight) {
    $item = Get-Item $weight
    $hash = Get-FileHash $weight -Algorithm SHA256
    $Manifest.files += [ordered]@{
      path = $weight
      bytes = $item.Length
      last_write_time = $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
      sha256 = $hash.Hash.ToLowerInvariant()
    }
  }
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $BackupDir "manifest.json") -Encoding UTF8

Write-Host "Starting clarity-focused MTR-UNet refiner training..."
Write-Host "Log: $LogPath"
Write-Host "Backup: $BackupDir"

$TrainArgs = @(
  "backend\train_reconstruction.py",
  "--masked", "database\processed_lfw",
  "--original", "database\lfw-deepfunneled",
  "--extra-original", "database\VGGFace2_train_cleaned\train",
  "--max-extra-originals", "30000",
  "--filter-extra-originals", "false",
  "--resume", "weights\degan_model_best_good_checkpoint.pth",
  "--use_refiner", "true",
  "--refiner-delta-scale", "0.75",
  "--epochs", "14",
  "--batch", "8",
  "--lr", "1e-4",
  "--workers", "0",
  "--synthetic-probability", "1.0",
  "--synthetic-mode", "opaque-panel",
  "--mask-weight", "13.0",
  "--full-weight", "0.2",
  "--identity-weight", "0.08",
  "--global-identity-weight", "0.25",
  "--edge-weight", "4.0",
  "--detail-weight", "5.0",
  "--structure-weight", "1.0",
  "--perceptual-weight", "1.6",
  "--adversarial-weight", "0.04",
  "--tv-weight", "0.0005",
  "--disc-lr", "5e-5",
  "--disc-start-epoch", "1"
)

if ($DryRun) {
  Write-Host "Dry run only. Command would be:"
  Write-Host "venv\Scripts\python.exe $($TrainArgs -join ' ')"
  exit 0
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& venv\Scripts\python.exe @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

if ($exitCode -ne 0) {
  throw "Clarity refiner training failed with exit code $exitCode. See log: $LogPath"
}

Write-Host "Clarity refiner training complete."
