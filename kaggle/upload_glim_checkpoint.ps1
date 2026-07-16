[CmdletBinding()]
param(
    [string]$DatasetSlug = "thestonedape/glim-zuco-checkpoint",
    [switch]$VersionExisting
)

$ErrorActionPreference = "Stop"

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$checkpointPath = Join-Path $workspaceRoot "GLIM\checkpoints\glim-zuco-epoch=199-step=49600.ckpt"
$metadataTemplate = Join-Path $PSScriptRoot "glim_checkpoint_dataset-metadata.json"
$stagePath = Join-Path $workspaceRoot "kaggle_upload\glim-zuco-checkpoint"
$stageCheckpoint = Join-Path $stagePath "glim-zuco-epoch=199-step=49600.ckpt"
$stageMetadata = Join-Path $stagePath "dataset-metadata.json"
$kaggleExe = Join-Path $workspaceRoot ".tools\kaggle-cli\Scripts\kaggle.exe"
$accessTokenPath = Join-Path $env:USERPROFILE ".kaggle\access_token"
$expectedSha256 = "25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba"

if (-not (Test-Path -LiteralPath $kaggleExe)) {
    throw "Kaggle CLI not found at $kaggleExe"
}
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "GLIM checkpoint not found at $checkpointPath"
}
if (-not (Test-Path -LiteralPath $metadataTemplate)) {
    throw "Dataset metadata template not found at $metadataTemplate"
}
if (-not $env:KAGGLE_API_TOKEN -and -not (Test-Path -LiteralPath $accessTokenPath)) {
    throw "No Kaggle token found. Set KAGGLE_API_TOKEN or create the user-profile .kaggle/access_token file."
}

$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointPath).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "GLIM checkpoint SHA-256 mismatch: $actualSha256"
}

$metadata = Get-Content -LiteralPath $metadataTemplate -Raw | ConvertFrom-Json
if ($metadata.id -ne $DatasetSlug) {
    throw "Metadata targets '$($metadata.id)', not '$DatasetSlug'."
}

New-Item -ItemType Directory -Path $stagePath -Force | Out-Null
Copy-Item -LiteralPath $metadataTemplate -Destination $stageMetadata -Force

if (Test-Path -LiteralPath $stageCheckpoint) {
    $stagedSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stageCheckpoint).Hash.ToLowerInvariant()
    if ($stagedSha256 -ne $expectedSha256) {
        throw "Existing staged checkpoint has the wrong SHA-256: $stagedSha256"
    }
} else {
    try {
        New-Item -ItemType HardLink -Path $stageCheckpoint -Target $checkpointPath | Out-Null
    } catch {
        Copy-Item -LiteralPath $checkpointPath -Destination $stageCheckpoint
    }
}

Write-Host "Verified checkpoint SHA-256 $expectedSha256"
Write-Host "Uploading private dataset $DatasetSlug"

if ($VersionExisting) {
    & $kaggleExe datasets version --path $stagePath --message "verified GLIM ZuCo checkpoint"
} else {
    & $kaggleExe datasets create --path $stagePath --private
}

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle checkpoint upload failed with exit code $LASTEXITCODE."
}

Write-Host "Upload completed: https://www.kaggle.com/datasets/$DatasetSlug"
