@echo off
title 视频批量无损转mp4
::set path=%path%

for /f "delims=" %%i in ('dir /b /a-d /s "*.y4m"') do ffmpeg -i "%%i" -y -qscale 0 -vcodec libx264 "%cd%\%%~ni.mp4" 

ping -n 5 127.0.0.1 >nul