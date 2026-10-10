$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskIndexRoot = Join-Path $taskWorkspace 'artifacts/thucnews_selected_full'
$taskLatest = Get-Content -LiteralPath (Join-Path $taskIndexRoot 'latest.json') -Raw | ConvertFrom-Json
$taskExperiment = Join-Path $taskWorkspace $taskLatest.experiment
$taskPlan = Get-Content -LiteralPath (Join-Path $taskExperiment 'index.json') -Raw | ConvertFrom-Json
Write-Output "实验状态：$($taskPlan.status)"
foreach ($taskMethod in $taskPlan.configuration.models) {
    $taskRun = Join-Path $taskWorkspace $taskPlan.runs.$taskMethod
    $taskStatePath = Join-Path $taskRun 'state.json'
    $taskSummaryPath = Join-Path $taskRun 'summary.json'
    $taskSealPath = Join-Path $taskRun 'seal.json'
    $taskLogPath = Join-Path $taskRun 'run.log'
    if (Test-Path -LiteralPath $taskSealPath) {
        $taskSummary = Get-Content -LiteralPath $taskSummaryPath -Raw | ConvertFrom-Json
        Write-Output "$taskMethod：训练完成 $($taskSummary.trained_epochs) 轮；选中第 $($taskSummary.selected_epoch) 轮；验证 Spearman=$($taskSummary.selection.best_score)"
    } elseif (Test-Path -LiteralPath $taskStatePath) {
        $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
        $taskNextEpoch = @($taskState.epochs).Count + 1
        Write-Output "$taskMethod：已保存 $(@($taskState.epochs).Count) 轮；最佳第 $($taskState.selection.best_epoch) 轮；验证 Spearman=$($taskState.selection.best_score)"
        if (Test-Path -LiteralPath $taskLogPath) {
            $taskLastLine = Get-Content -LiteralPath $taskLogPath -Tail 1
            if ($taskLastLine -match 'at ([0-9.]+)% examples') {
                Write-Output "第 $taskNextEpoch 轮已遍历约 $($Matches[1])% 的句段（非整个实验进度）。"
            } else {
                Write-Output $taskLastLine
            }
        }
    } elseif (Test-Path -LiteralPath $taskLogPath) {
        Write-Output "$taskMethod：已启动，尚无完整轮次检查点。"
        Get-Content -LiteralPath $taskLogPath -Tail 1
    } else {
        Write-Output "$taskMethod：排队等待。"
    }
    if (Test-Path -LiteralPath $taskLogPath) { Write-Output "日志：$taskLogPath" }
}
Write-Output "独立核验：$(Test-Path -LiteralPath (Join-Path $taskExperiment 'verification.json'))"
