@echo off
title FaceCam + Cloudflare Tunnel
echo =========================================
echo  Bat script start Flask + Tunnel (facecam)
echo =========================================
echo.

:: Bước 1: chạy Flask app (đúng link tới file py của bạn)
start "Flask App" cmd /k python C:\Users\Admin\Documents\GitHub\MTCNN-and-Facenet-Face-Recognition-System-for-Arduino-and-Esp32-Control\src\face_rec_web.py --no-gui --no-camera

:: Đợi 5 giây để Flask khởi động
timeout /t 5 /nobreak >nul

:: Bước 2: chạy Cloudflare Tunnel
start "Cloudflare Tunnel" cmd /k cloudflared tunnel run facecam

echo.
echo Flask app và Tunnel đã chạy.
echo Mở domain: https://nngocsonk3.id.vn/
echo.
pause
