@echo off
title 视频批量无损转y4m
::set path=%path%

for /f "delims=" %%i in ('dir /b /a-d /s "*.mp4"') do ffmpeg -i "%%i" -y -qscale 0 -pix_fmt yuv420p "%cd%\%%~ni.y4m" 

ping -n 5 127.0.0.1 >nul