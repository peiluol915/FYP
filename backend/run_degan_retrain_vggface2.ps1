$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$python = "venv\Scripts\python.exe"
$masked = "database\lfw_15may"
$original = "database\lfw-deepfunneled"
$vggface2CleanTrain = "database\VGGFace2_train_cleaned\train"
$resume = "weights\degan_model_best.pth"
$maxExtraOriginals = 30000
$batchSize = 8
$workerCount = 0
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $ProjectRoot "outputs\degan_vggface2_retrain_$stamp.log"
$backupDir = Join-Path $ProjectRoot "weights\backups\degan_before_vggface2_retrain_$stamp"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path $vggface2CleanTrain)) {
  throw "Cleaned VGGFace2 train split was not found: $vggface2CleanTrain"
}

New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
foreach ($path in @("weights\degan_model_best.pth", "weights\degan_model_final.pth", "weights\degan_disc_final.pth")) {
  if (Test-Path $path) {
    Copy-Item $path (Join-Path $backupDir (Split-Path $path -Leaf)) -Force
  }
}

Write-Host "Starting DE-GAN retraining with cleaned VGGFace2 extra originals."
Write-Host "Backup: $backupDir"
Write-Host "Log: $logPath"

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & $python "backend\train_reconstruction.py" `
    --masked $masked `
    --original $original `
    --extra-original $vggface2CleanTrain `
    --max-extra-originals $maxExtraOriginals `
    --filter-extra-originals false `
    --resume $resume `
    --epochs 20 `
    --batch $batchSize `
    --lr 2e-5 `
    --workers $workerCount `
    --synthetic-probability 0.98 `
    --synthetic-mode opaque-panel `
    --mask-weight 14.0 `
    --full-weight 0.5 `
    --identity-weight 0.05 `
    --global-identity-weight 0.25 `
    --edge-weight 2.5 `
    --structure-weight 2.0 `
    --perceptual-weight 1.8 `
    --adversarial-weight 0.04 `
    --tv-weight 0.008 `
    --detail-weight 2.5 `
    --disc-lr 5e-5 `
    --disc-start-epoch 1 `
    --mask-threshold 0.08 `
    --use-refiner false `
    2>&1 | Tee-Object -FilePath $logPath
  $trainingExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $previousErrorActionPreference
}

if ($trainingExitCode -ne 0) {
  throw "DE-GAN VGGFace2 retraining failed with exit code $trainingExitCode. See log: $logPath"
}

Write-Host "DE-GAN VGGFace2 retraining complete."
