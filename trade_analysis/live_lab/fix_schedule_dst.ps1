# Fix the "LiveLab-QQQ-SPY" scheduled task for the end of daylight saving (2026-11-01).
#
# Run on the LAPTOP, in normal (not admin) PowerShell, from the repo folder:
#   powershell -ExecutionPolicy Bypass -File trade_analysis\live_lab\fix_schedule_dst.ps1
#
# WHY
# Arizona stays on MST (UTC-7) all year. New York moves from UTC-4 to UTC-5 on 2026-11-01,
# so the task's fixed 05:30 MST start becomes 07:30 ET instead of 08:30 ET.
# The task repeats for 8 hours with StopAtDurationEnd = True. In winter that window ends at
# 13:30 MST = 15:30 ET, and StopAtDurationEnd tells Windows to END THE RUNNING INSTANCE at
# that moment: 26 minutes before the 15:56 flatten, the 16:00 daily summary and the commit.
#
# WHAT THIS CHANGES (and nothing else)
#   StopAtDurationEnd  True -> False  never kill a running session at the window end
#   Duration           8 h  -> 10 h   repeats until 15:30 MST = 17:30 ET winter, 18:30 ET
#                                     summer, so the after-close stranded pass at ~16:30 ET
#                                     still happens in winter
# Start time, days, action, WakeToRun, IgnoreNew, 12 h time limit, restarts: untouched.
# Extra evening runs are harmless: a second autostart exits if one is already supervising
# (autostart.py, SingleInstance), and after 15:30 ET one only runs resolve_stranded and exits.
# The 12 h ExecutionTimeLimit still caps a stuck instance at 17:30 MST.

$ErrorActionPreference = 'Stop'
$name = 'LiveLab-QQQ-SPY'

$backup = Join-Path $env:USERPROFILE ("Documents\{0}_backup_{1:yyyyMMdd_HHmmss}.xml" -f $name, (Get-Date))
Export-ScheduledTask -TaskName $name | Out-File -FilePath $backup -Encoding utf8
Write-Host "backup saved: $backup"
Write-Host "  (undo: Register-ScheduledTask -TaskName $name -Xml (Get-Content '$backup' -Raw) -Force)"

$task = Get-ScheduledTask -TaskName $name
if ($task.Triggers.Count -ne 1) { throw "expected 1 trigger, found $($task.Triggers.Count); not touching it" }

$rep = $task.Triggers[0].Repetition
Write-Host ("before: start {0}  every {1}  for {2}  StopAtDurationEnd={3}" -f `
    $task.Triggers[0].StartBoundary, $rep.Interval, $rep.Duration, $rep.StopAtDurationEnd)

$rep.Duration = 'PT10H'
$rep.StopAtDurationEnd = $false
Set-ScheduledTask -InputObject $task | Out-Null

$after = (Get-ScheduledTask -TaskName $name).Triggers[0]
Write-Host ("after:  start {0}  every {1}  for {2}  StopAtDurationEnd={3}" -f `
    $after.StartBoundary, $after.Repetition.Interval, $after.Repetition.Duration, $after.Repetition.StopAtDurationEnd)

if ($after.Repetition.Duration -ne 'PT10H' -or $after.Repetition.StopAtDurationEnd) {
    throw "the change did not stick; restore from $backup"
}
Write-Host "OK"
