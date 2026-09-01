@echo off
REM Unattended launcher for the live paper-trading lab.
REM Registered with Windows Task Scheduler as "LiveLab-QQQ-SPY".
REM Started early on purpose: autostart.py sleeps until 09:20 EXCHANGE time, so the
REM MST<->ET offset changing with DST cannot drift the start.
cd /d "C:\Users\chaitanyakharche\Documents\stock"
"C:\Users\chaitanyakharche\.pyenv\pyenv-win\versions\3.12.10\python.exe" -m trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal
