param(
    [switch]$DryRun,
    [int]$StartAt = 1,
    [int]$Gpu = 0,
    [int]$EvalGpu = 0,
    [string]$Python = "C:\Users\Administrator\.conda\envs\deeplearning\python.exe",
    [string]$RunLogName = "run_six_cases.log"
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$logDir = "logs\six_cases"
$runLog = Join-Path $logDir $RunLogName

New-Item -ItemType Directory -Force $logDir | Out-Null

$runs = @(
    @{
        Index = 1
        Config = "config\custom_sweep\run_1_pic_lb100_5_ulb250_5_exp.yaml"
        SaveName = "run_1_pic_lb100_5_ulb250_5_exp"
    },
    @{
        Index = 2
        Config = "config\custom_sweep\run_2_pic_lb100_10_ulb250_10_exp.yaml"
        SaveName = "run_2_pic_lb100_10_ulb250_10_exp"
    },
    @{
        Index = 3
        Config = "config\custom_sweep\run_3_pic_lb100_5_ulb315_1.0_exp.yaml"
        SaveName = "run_3_pic_lb100_5_ulb315_1.0_exp"
    },
    @{
        Index = 4
        Config = "config\custom_sweep\run_4_pic_lb100_10_ulb315_1.0_exp.yaml"
        SaveName = "run_4_pic_lb100_10_ulb315_1.0_exp"
    },
    @{
        Index = 5
        Config = "config\custom_sweep\run_5_pic_lb100_5_ulb250_-5_pxe.yaml"
        SaveName = "run_5_pic_lb100_5_ulb250_-5_pxe"
    },
    @{
        Index = 6
        Config = "config\custom_sweep\run_6_pic_lb100_10_ulb250_-10_pxe.yaml"
        SaveName = "run_6_pic_lb100_10_ulb250_-10_pxe"
    },
    @{
        Index = 7
        Config = "config\custom_sweep\run_7_pic_lb100_50_ulb250_50_exp.yaml"
        SaveName = "run_7_pic_lb100_50_ulb250_50_exp"
    },
    @{
        Index = 8
        Config = "config\custom_sweep\run_8_pic_lb100_100_ulb250_100_exp.yaml"
        SaveName = "run_8_pic_lb100_100_ulb250_100_exp"
    }
) | Where-Object { $_.Index -ge $StartAt }

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] custom sweep started start_at=$StartAt gpu=$Gpu eval_gpu=$EvalGpu python=$Python" | Tee-Object -FilePath $runLog

foreach ($run in $runs) {
    $config = $run.Config
    $saveName = $run.SaveName
    $checkpoint = Join-Path "saved_models" (Join-Path $saveName "model_best.pth")
    $trainLog = Join-Path $logDir "$saveName.train.log"
    $evalLog = Join-Path $logDir "$saveName.eval.log"

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START train $config" | Tee-Object -FilePath $runLog -Append
    if ($DryRun) {
        "DRYRUN: & `"$Python`" train.py --c `"$config`"" | Tee-Object -FilePath $runLog -Append
    }
    else {
        $cmd = "`"$Python`" train.py --c `"$config`" 2>&1"
        & cmd.exe /d /c $cmd | Tee-Object -FilePath $trainLog
        if ($LASTEXITCODE -ne 0) {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED train $config exit=$LASTEXITCODE" | Tee-Object -FilePath $runLog -Append
            exit $LASTEXITCODE
        }
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START eval $config checkpoint=$checkpoint" | Tee-Object -FilePath $runLog -Append
    if ($DryRun) {
        "DRYRUN: & `"$Python`" eval_confusion_matrix_new.py --c `"$config`" --load_path `"$checkpoint`" --gpu $EvalGpu" | Tee-Object -FilePath $runLog -Append
    }
    else {
        if (-not (Test-Path $checkpoint)) {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED eval checkpoint missing: $checkpoint" | Tee-Object -FilePath $runLog -Append
            exit 1
        }

        $cmd = "`"$Python`" eval_confusion_matrix_new.py --c `"$config`" --load_path `"$checkpoint`" --gpu $EvalGpu 2>&1"
        & cmd.exe /d /c $cmd | Tee-Object -FilePath $evalLog
        if ($LASTEXITCODE -ne 0) {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED eval $config exit=$LASTEXITCODE" | Tee-Object -FilePath $runLog -Append
            exit $LASTEXITCODE
        }
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $config" | Tee-Object -FilePath $runLog -Append
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] custom sweep finished" | Tee-Object -FilePath $runLog -Append
