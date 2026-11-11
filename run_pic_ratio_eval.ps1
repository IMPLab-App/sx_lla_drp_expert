param(
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$python = "C:\Users\Administrator\.conda\envs\deeplearning\python.exe"
$logDir = "logs\ratio_sweep"
$runLog = Join-Path $logDir "eval_all.log"

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

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio eval started gpu=$Gpu" | Tee-Object -FilePath $runLog

foreach ($config in $configs) {
    if (-not (Test-Path $config)) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] missing config: $config" | Tee-Object -FilePath $runLog -Append
        exit 1
    }

    $name = [System.IO.Path]::GetFileNameWithoutExtension($config)
    $runDir = Join-Path "saved_models" $name
    $ckpt = Join-Path $runDir "model_best.pth"
    $logPath = Join-Path $logDir "$name.eval.log"

    if (-not (Test-Path $ckpt)) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] missing checkpoint: $ckpt" | Tee-Object -FilePath $runLog -Append
        exit 1
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] EVAL $config" | Tee-Object -FilePath $runLog -Append

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cmd = "`"$python`" eval_confusion_matrix.py --c `"$config`" --load_path `"$ckpt`" --gpu $Gpu 2>&1"
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

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio eval finished" | Tee-Object -FilePath $runLog -Append
