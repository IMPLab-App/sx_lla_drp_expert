param(
    [switch]$DryRun,
    [int]$StartAt = 1,
    [string]$RunLogName = "run_all.log"
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$env:CUDA_VISIBLE_DEVICES = "0"
$python = "C:\Users\Administrator\.conda\envs\deeplearning\python.exe"
$logDir = "logs\ratio_sweep"
$runLog = Join-Path $logDir $RunLogName

New-Item -ItemType Directory -Force $logDir | Out-Null

$configs = @(
    "config\ratio_sweep\001-fixmatch_metaexpert_pic_lb900_50_ulb1800_50_0.0_2.yaml",
    "config\ratio_sweep\002-fixmatch_metaexpert_pic_lb900_100_ulb1800_100_0.0_2.yaml",
    "config\ratio_sweep\003-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2.yaml",
    "config\ratio_sweep\004-fixmatch_metaexpert_pic_lb900_200_ulb1800_200_0.0_2.yaml",
    "config\ratio_sweep\005-fixmatch_metaexpert_pic_lb900_50_ulb1800_-50_0.0_2.yaml",
    "config\ratio_sweep\006-fixmatch_metaexpert_pic_lb900_100_ulb1800_-100_0.0_2.yaml",
    "config\ratio_sweep\007-fixmatch_metaexpert_pic_lb900_150_ulb1800_-150_0.0_2.yaml",
    "config\ratio_sweep\008-fixmatch_metaexpert_pic_lb900_200_ulb1800_-200_0.0_2.yaml"
)

$configs = $configs | Where-Object {
    $match = [regex]::Match([System.IO.Path]::GetFileName($_), '^\d+')
    $match.Success -and ([int]$match.Value -ge $StartAt)
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio sweep started start_at=$StartAt" | Tee-Object -FilePath $runLog

if (-not $DryRun) {
    $indexCacheDir = "data\raw_signal\labeled_idx"
    if (Test-Path $indexCacheDir) {
        Get-ChildItem -Path $indexCacheDir -Filter "*.npy" -File | Remove-Item -Force
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] cleared stale index cache: $indexCacheDir\*.npy" | Tee-Object -FilePath $runLog -Append
    }
}

foreach ($config in $configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($config)
    $logPath = Join-Path $logDir "$name.log"

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $config" | Tee-Object -FilePath $runLog -Append
    if ($DryRun) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DRYRUN $config" | Tee-Object -FilePath $runLog -Append
        continue
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cmd = "`"$python`" train.py --c `"$config`" 2>&1"
        & cmd.exe /d /c $cmd | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED $config exit=$exitCode" | Tee-Object -FilePath $runLog -Append
        exit $exitCode
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $config" | Tee-Object -FilePath $runLog -Append
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio sweep finished" | Tee-Object -FilePath $runLog -Append
