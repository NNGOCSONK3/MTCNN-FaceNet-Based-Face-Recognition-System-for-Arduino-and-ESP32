# Đường dẫn thay cho phù hợp
$cloudflared = "$env:USERPROFILE\.cloudflared\cloudflared.exe"
$config = "$env:USERPROFILE\.cloudflared\config.yml"
$venvPython = "python"  # hoặc đường dẫn python cụ thể

# 1) Start Flask
Start-Process -NoNewWindow -FilePath $venvPython -ArgumentList "src\face_rec_web.py","--esp32_ip","auto","--esp32_port","8088","--no-gui"

Start-Sleep -Seconds 3  # đợi Flask mở port

# 2) Start Cloudflare Tunnel (named)
Start-Process -NoNewWindow -FilePath $cloudflared -ArgumentList "tunnel","--config",$config,"run","facecam"
