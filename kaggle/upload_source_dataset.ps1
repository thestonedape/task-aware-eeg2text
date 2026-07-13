[CmdletBinding()]
param(
    [string]$DatasetSlug = "thestonedape/task-aware-eeg2text-source",
    [string]$VersionMessage = "add-frozen-dataframe-source",
    [switch]$StopExistingUpload
)

$ErrorActionPreference = "Stop"

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stagePath = Join-Path $workspaceRoot "kaggle_upload\task-aware-eeg2text-source"
$sourcePath = Join-Path $stagePath "zuco_eeg_label_8variants.df"
$metadataPath = Join-Path $stagePath "dataset-metadata.json"
$pidPath = Join-Path $workspaceRoot "kaggle_upload\version_pid.txt"
$kaggleExe = Join-Path $workspaceRoot ".tools\kaggle-cli\Scripts\kaggle.exe"
$accessTokenPath = Join-Path $HOME ".kaggle\access_token"

if (-not (Test-Path -LiteralPath $kaggleExe)) {
    throw "Kaggle CLI not found at $kaggleExe"
}
if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Source dataframe not found at $sourcePath"
}
if (-not (Test-Path -LiteralPath $metadataPath)) {
    throw "Dataset metadata not found at $metadataPath"
}
if (-not $env:KAGGLE_API_TOKEN -and -not (Test-Path -LiteralPath $accessTokenPath)) {
    throw "No Kaggle token found. Set KAGGLE_API_TOKEN or create ~/.kaggle/access_token."
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.id -ne $DatasetSlug) {
    throw "dataset-metadata.json targets '$($metadata.id)', not '$DatasetSlug'."
}

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    if ($oldPid -match '^\d+$') {
        $oldProcess = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($oldProcess -and $oldProcess.ProcessName -like "kaggle*") {
            if (-not $StopExistingUpload) {
                throw "Kaggle upload PID $oldPid is already running. Re-run with -StopExistingUpload to cancel it and restart on the faster connection."
            }
            Write-Host "Stopping previous Kaggle upload PID $oldPid..."
            Stop-Process -Id ([int]$oldPid) -Force
            Start-Sleep -Seconds 2
        }
    }
}

$sizeGiB = [math]::Round((Get-Item -LiteralPath $sourcePath).Length / 1GB, 2)
Write-Host "Uploading $sizeGiB GiB to private dataset $DatasetSlug"
Write-Host "Keep this PowerShell window open. Kaggle CLI does not reliably resume an interrupted file upload."

& $kaggleExe datasets version `
    --path $stagePath `
    --message $VersionMessage `
    --keep-tabular

if ($LASTEXITCODE -ne 0) {
    throw "Kaggle dataset upload failed with exit code $LASTEXITCODE."
}

Write-Host "Upload completed: https://www.kaggle.com/datasets/$DatasetSlug"
