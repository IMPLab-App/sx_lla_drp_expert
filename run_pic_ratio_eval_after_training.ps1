param(
    [int]$Gpu = 0,
    [int]$PollSeconds = 60,
    [string]$TrainLogName = "run_all.log"
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$logDir = "logs\ratio_sweep"
$trainLog = Join-Path $logDir $TrainLogName
$watchLog = Join-Path $logDir "eval_after_training.log"

New-Item -ItemType Directory -Force $logDir | Out-Null

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] waiting for ratio sweep to finish" | Tee-Object -FilePath $watchLog

while ($true) {
    if (Test-Path $trainLog) {
        $tail = Get-Content $trainLog -Tail 40
        if ($tail -match "FAILED") {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] training failed; skip evaluation" | Tee-Object -FilePath $watchLog -Append
            exit 1
        }
        if ($tail -match "ratio sweep finished") {
            break
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] training finished; starting ratio eval" | Tee-Object -FilePath $watchLog -Append

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run_pic_ratio_eval.ps1" -Gpu $Gpu *>&1 |
    Tee-Object -FilePath $watchLog -Append

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio eval failed exit=$exitCode" | Tee-Object -FilePath $watchLog -Append
    exit $exitCode
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ratio eval after training finished" | Tee-Object -FilePath $watchLog -Append
