# -*- coding: utf-8 -*-
"""
Face recognition + ESP32 TCP control + Flask MJPEG web + robust auto-discovery.

Endpoints:
- /                 : UI mobile (Logo + trạng thái + F5 + Cam + LED1/2/3 + Door + Fan + DHT11)
- /video_feed       : MJPEG stream
- /status           : trạng thái kết nối ESP
- /telemetry        : JSON cảm biến/ACK mới nhất
- /logo             : trả về /src/images/Logo.png

Auto-discovery:
- Nếu --esp32_ip=auto: quét subnet /24, ưu tiên ARP; chỉ nhận thiết bị gửi '{"hello":"esp32"}'.
- Không hiển thị nút START; nhưng tự động gửi 'start' đúng 1 lần khi vừa kết nối thành công.
"""

import os
import time
import socket
import threading
import pickle
import argparse
import re
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import imutils
from imutils.video import VideoStream

# TensorFlow v1 style (facenet + MTCNN)
import tensorflow as tf
import facenet
from align import detect_face as detect_face
from sklearn.svm import SVC

from flask import Flask, Response, make_response, jsonify, request, send_file
from flask_cors import CORS

# ================= Global flags =================
AUTO_START_ON_CONNECT = True  # Gửi "start" một lần sau khi kết nối ESP

# ================= Flask & stream =================
app = Flask(__name__)
CORS(app)

_frame_lock = threading.Lock()
_latest_jpeg = None
_jpeg_quality = 80

def update_stream_frame(frame_bgr):
    global _latest_jpeg
    ok, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_quality])
    if ok:
        with _frame_lock:
            _latest_jpeg = jpeg.tobytes()

def _mjpeg_generator():
    boundary = b'--frame'
    while True:
        with _frame_lock:
            buf = _latest_jpeg
        if buf is None:
            time.sleep(0.03)
            yield boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n'
            continue
        yield boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= ESP state & telemetry =================
_esp = {"sock": None, "connected": False, "ip": "", "port": 0, "last_error": ""}
_esp_lock = threading.Lock()

_last_telemetry = {}
_last_ack = {}
_last_t_lock = threading.Lock()

@app.route('/status')
def status():
    with _esp_lock:
        return jsonify({
            "esp_connected": _esp["connected"],
            "ip": _esp["ip"],
            "port": _esp["port"],
            "last_error": _esp["last_error"],
        })

@app.route('/telemetry')
def telemetry():
    with _last_t_lock:
        out = dict(_last_telemetry)
        if _last_ack:
            out["ack"] = _last_ack
        return jsonify(out)

@app.route('/logo')
def logo():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, 'src', 'images', 'logo_mobile.png')
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    return Response(status=404)

def _set_esp_state(sock, connected, ip, port, last_error=""):
    with _esp_lock:
        _esp.update({"sock": sock, "connected": connected, "ip": ip or "", "port": port, "last_error": last_error})

# ================= Unified send =================
def send_to_esp_raw(cmd: str):
    """Gửi chuỗi lệnh thô tới ESP qua TCP (tự thêm newline)."""
    if not cmd.endswith("\n"):
        cmd += "\n"
    with _esp_lock:
        s, ok, ip, port = _esp["sock"], _esp["connected"], _esp["ip"], _esp["port"]
    if s and ok:
        try:
            s.send(cmd.encode("utf-8"))
            return True, "sent"
        except socket.error as e:
            _set_esp_state(None, False, ip, port, str(e))
            return False, f"socket error: {e}"
    return False, "esp not connected"

def send_command(kind: str, value=None, index: int = None):
    """
    Hỗ trợ:
      - light id=1..3, value in {on,off}  -> "light1:on"
      - fan on/off                        -> "fan:on/off"
      - door open/close                   -> "door:open/close"
      - servo:<angle>                     -> "servo:0..180" (tương thích)
      - start                             -> "start" (một lần khi connect hoặc face-rec)
      - ping                              -> "ping"
      - raw:<string>                      -> gửi nguyên văn
      - led on/off                        -> alias light1 on/off
    """
    kind = (kind or "").lower().strip()

    if kind == "light":
        if index not in (1,2,3):
            return False, "invalid light index"
        state = str(value).lower()
        if state not in ("on","off"):
            return False, "invalid light state"
        return send_to_esp_raw(f"light{index}:{state}")

    if kind == "fan":
        state = str(value).lower()
        if state not in ("on","off"):
            return False, "invalid fan state"
        return send_to_esp_raw(f"fan:{state}")

    if kind == "door":
        state = str(value).lower()
        if state not in ("open","close"):
            return False, "invalid door state"
        return send_to_esp_raw(f"door:{state}")

    if kind == "servo":
        try:
            angle = int(value)
        except Exception:
            return False, "invalid servo angle"
        angle = max(0, min(180, angle))
        return send_to_esp_raw(f"servo:{angle}")

    if kind == "led":
        state = str(value).lower()
        if state not in ("on","off"):
            return False, "invalid led state"
        return send_to_esp_raw(f"light1:{state}")

    if kind == "start":
        return send_to_esp_raw("start")

    if kind == "ping":
        return send_to_esp_raw("ping")

    if kind == "raw":
        if not isinstance(value, str) or not value:
            return False, "invalid raw command"
        return send_to_esp_raw(value)

    return False, "unknown command"

# ================= APIs =================
def _json_bad(msg): return ({"ok": False, "msg": msg}, 400)

@app.route('/api/light1', methods=['POST'])
def api_light1():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("on","off"): return _json_bad("state must be on/off")
    ok, msg = send_command("light", state, 1)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

@app.route('/api/light2', methods=['POST'])
def api_light2():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("on","off"): return _json_bad("state must be on/off")
    ok, msg = send_command("light", state, 2)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

@app.route('/api/light3', methods=['POST'])
def api_light3():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("on","off"): return _json_bad("state must be on/off")
    ok, msg = send_command("light", state, 3)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

@app.route('/api/door', methods=['POST'])
def api_door():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("open","close"): return _json_bad("state must be open/close")
    ok, msg = send_command("door", state)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

@app.route('/api/fan', methods=['POST'])
def api_fan():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("on","off"): return _json_bad("state must be on/off")
    ok, msg = send_command("fan", state)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

# Tương thích cũ:
@app.route('/api/led', methods=['POST'])
def api_led():
    state = str((request.get_json(silent=True) or {}).get("state","")).lower()
    if state not in ("on","off"): return _json_bad("state must be on/off")
    ok, msg = send_command("led", state)
    return ({"ok": ok, "msg": msg}, 200 if ok else 503)

@app.route('/api/servo', methods=['POST'])
def api_servo():
    data = request.get_json(silent=True) or {}
    try:
        angle = int(data.get("angle", 90))
    except Exception:
        angle = 90
    angle = max(0, min(180, angle))
    ok, msg = send_command("servo", angle)
    return ({"ok": ok, "msg": msg, "angle": angle}, 200 if ok else 503)

# ================= Helper: LAN IP & Flask start =================
def _get_local_ip():
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    return ip

def start_flask_server():
    lan_ip = _get_local_ip()
    print("Web xem video:")
    print("  • Trên máy này:   http://localhost:5000/")
    print(f"  • Thiết bị khác:  http://{lan_ip}:5000/")
    app.run(host='0.0.0.0', port=5000, threaded=True)

# ================= Discovery =================
HELLO_SIGNATURE = '"hello":"esp32"'

def _cidr_hosts(base_ip):
    parts = base_ip.split(".")
    subnet = ".".join(parts[:3])
    return [f"{subnet}.{i}" for i in range(2, 255)]

def _arp_candidates(subnet_prefix):
    cands = []
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, encoding="utf-8", errors="ignore")
        for line in out.splitlines():
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if m:
                ip = m.group(1)
                if ip.startswith(subnet_prefix + ".") and not ip.endswith(".1"):
                    cands.append(ip)
    except Exception:
        pass
    seen, res = set(), []
    for x in cands:
        if x not in seen:
            res.append(x); seen.add(x)
    return res

def _probe_esp(ip, port, timeout=0.45):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        try:
            s.settimeout(timeout)
            data = s.recv(256)
        except socket.timeout:
            data = b""
        txt = (data or b"").decode("utf-8", "ignore")
        s.close()
        return (HELLO_SIGNATURE in txt)
    except Exception:
        try:
            s.close()
        except:
            pass
        return False

def discover_esp_ip(port=8088, timeout_per_host=0.45, max_workers=64):
    local_ip = _get_local_ip()
    parts = local_ip.split(".")
    if len(parts) != 4:
        return None
    prefix = ".".join(parts[:3])

    arp = _arp_candidates(prefix)
    all_hosts = _cidr_hosts(local_ip)
    seen = set(arp)
    candidates = arp + [h for h in all_hosts if h not in seen and not h.endswith(".1")]

    print(f"[DISCOVERY] Quét {prefix}.0/24 tìm ESP (port {port}) ...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2ip = {ex.submit(_probe_esp, ip, port, timeout_per_host): ip for ip in candidates}
        for fut in as_completed(fut2ip):
            ip = fut2ip[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                print(f"[DISCOVERY] Xác thực ESP32 tại {ip}:{port}")
                return ip
    print("[DISCOVERY] Chưa tìm thấy ESP trong subnet.")
    return None

# ================= ESP reconnect & telemetry =================
_target_lock = threading.Lock()
_target = {"ip": None, "port": 8088}
_sent_start_after_connect = False

def esp_reconnector(target_ref, interval=8):
    global _sent_start_after_connect
    while True:
        with _esp_lock:
            ok = _esp["connected"]
        if not ok:
            with _target_lock:
                ip = target_ref["ip"]; port = target_ref["port"]
            if not ip:
                time.sleep(interval)
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((ip, port))

                # Handshake
                hello = b""
                try:
                    s.settimeout(1.0)
                    hello = s.recv(256)
                except socket.timeout:
                    pass
                txt = (hello or b"").decode("utf-8", "ignore")
                if HELLO_SIGNATURE not in txt:
                    s.close()
                    raise socket.error("handshake failed (not esp32)")

                s.settimeout(None)
                print(f"[ESP] Kết nối thành công {ip}:{port}")
                _set_esp_state(s, True, ip, port, "")
                _sent_start_after_connect = False  # reset cờ cho lần kết nối mới

                # Auto-START đúng 1 lần sau khi kết nối
                if AUTO_START_ON_CONNECT and not _sent_start_after_connect:
                    ok2, msg2 = send_command("start")
                    if ok2:
                        print("[ESP] Auto-START đã gửi sau khi kết nối.")
                        _sent_start_after_connect = True
                    else:
                        print("[ESP] Auto-START lỗi:", msg2)

            except socket.error as e:
                _set_esp_state(None, False, ip, port, str(e))
        time.sleep(interval)

def telemetry_reader():
    global _last_ack
    buf = b""
    while True:
        with _esp_lock:
            s = _esp["sock"] if _esp["connected"] else None
            ip = _esp["ip"]; port = _esp["port"]
        if not s:
            time.sleep(0.5)
            continue
        try:
            s.settimeout(0.2)
            chunk = s.recv(4096)
            if not chunk:
                _set_esp_state(None, False, ip, port, "peer closed")
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8", "ignore"))
                    with _last_t_lock:
                        if isinstance(data, dict) and "ack" in data:
                            _last_ack = data
                        else:
                            _last_telemetry.update(data)
                except Exception:
                    # bỏ qua dòng không phải JSON
                    pass
        except socket.timeout:
            pass
        except Exception as e:
            _set_esp_state(None, False, ip, port, str(e))
            time.sleep(0.5)

def rediscover_loop():
    while True:
        with _esp_lock:
            connected = _esp["connected"]
        with _target_lock:
            have_ip = bool(_target["ip"])
            port = _target["port"]
        if (not connected) and (not have_ip):
            ip_found = discover_esp_ip(port=port)
            if ip_found:
                print(f"[DISCOVERY] Cập nhật IP ESP32: {ip_found}:{port}")
                with _target_lock:
                    _target["ip"] = ip_found
                _set_esp_state(None, False, ip_found, port, "Chưa kết nối")
        time.sleep(8)

# ================= UI (mobile-like) =================
HTML_TEMPLATE = r"""
<!doctype html><html lang="vi">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>FaceCam</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden}
body{font-family:system-ui,Segoe UI,Roboto;background:#fff;color:#111}
.app{display:flex;flex-direction:column;gap:10px;height:100vh;padding:10px}

/* Logo trên cùng, chiếm riêng 1 hàng */
.logoTop{display:flex;justify-content:center;align-items:center}
.logoTop img{height:46px;width:auto}

/* Thanh trạng thái đặt DƯỚI logo, gọn nhẹ */
.statusBar{display:grid;grid-template-columns:1fr auto 1fr;align-items:center}
.badge{justify-self:start;padding:6px 10px;border:1px solid #e5e7eb;border-radius:999px;background:#fafafa;font-size:13px;white-space:nowrap}
.ok{background:#e8f7ef;border-color:#b7e3c7;color:#066a36}
.warn{background:#fff3f3;border-color:#ffd6d6;color:#991b1b}
.refresh{justify-self:end;padding:6px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#f3f4f6;font-size:13px}

/* Phần còn lại giữ nguyên bố cục */
.main{display:grid;grid-template-rows:auto auto 1fr;gap:10px;min-height:0}
.cam{min-height:0}
.cam img{width:100%;height:44vh;object-fit:contain;border:1px solid #e5e7eb;border-radius:12px;background:#fff}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.tile{border:1px solid #e5e7eb;border-radius:12px;background:#fff;padding:10px;display:flex;flex-direction:column;gap:8px;min-width:0}
.tile h4{margin:0 0 2px 0;font-size:14px;color:#111}
.row{display:flex;gap:8px;flex-wrap:wrap}
.btn{flex:1;padding:10px 12px;border:1px solid #d1d5db;border-radius:10px;background:#f3f4f6;font-size:14px}
.btn:active{transform:scale(0.98)}
.hint{font-size:12px;opacity:.85}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;font-size:14px}
.value{font-weight:700}
*::-webkit-scrollbar{display:none}
*{scrollbar-width:none}
@media (max-width:640px){.cam img{height:42vh}.btn{font-size:13px}}
</style>

<div class="app">
  <!-- LOGO trên cùng -->
  <div class="logoTop">
    <img src="/logo" alt="Logo" onerror="this.style.display='none'">
  </div>

  <!-- Thanh trạng thái ESP + nút F5 đặt dưới logo -->
  <div class="statusBar">
    <div id="esp" class="badge warn">Đang chờ kết nối</div>
    <div></div>
    <button class="refresh" onclick="location.reload()">Làm mới</button>
  </div>

  <div class="main">
    <div class="cam"><img src="/video_feed" alt="Camera"></div>

    <!-- Hàng 1: LED 1/2/3 -->
    <div class="grid3">
      <div class="tile">
        <h4>LED 1</h4>
        <div class="row">
          <button class="btn" onclick="light(1,'on')">Bật</button>
          <button class="btn" onclick="light(1,'off')">Tắt</button>
        </div>
        <div id="m_led1" class="hint"></div>
      </div>
      <div class="tile">
        <h4>LED 2</h4>
        <div class="row">
          <button class="btn" onclick="light(2,'on')">Bật</button>
          <button class="btn" onclick="light(2,'off')">Tắt</button>
        </div>
        <div id="m_led2" class="hint"></div>
      </div>
      <div class="tile">
        <h4>LED 3</h4>
        <div class="row">
          <button class="btn" onclick="light(3,'on')">Bật</button>
          <button class="btn" onclick="light(3,'off')">Tắt</button>
        </div>
        <div id="m_led3" class="hint"></div>
      </div>
    </div>

    <!-- Hàng 2: Door / Fan / DHT11 -->
    <div class="grid3">
      <div class="tile">
        <h4>Cửa</h4>
        <div class="row">
          <button class="btn" onclick="door('open')">Mở</button>
          <button class="btn" onclick="door('close')">Đóng</button>
        </div>
        <div id="m_door" class="hint"></div>
      </div>
      <div class="tile">
        <h4>Quạt</h4>
        <div class="row">
          <button class="btn" onclick="fan('on')">Bật</button>
          <button class="btn" onclick="fan('off')">Tắt</button>
        </div>
        <div id="m_fan" class="hint"></div>
      </div>
      <div class="tile">
        <h4>DHT11</h4>
        <div class="kv">
            <div>Nhiệt độ</div><div>27°C</div>
             <div>Độ ẩm</div><div>87%</div>
        </div>
        <div id="m_dht" class="hint"></div>
      </div>
    </div>
  </div>
</div>

<script>
function msg(id, text){const el=document.getElementById(id); if(el) el.textContent=text||'';}
async function post(url, body){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  try{ return await r.json(); }catch(e){ return {ok:false,msg:'bad json'} }
}
async function light(i, state){const j=await post('/api/light'+i,{state}); msg('m_led'+i, j.ok?'OK':'ERR: '+(j.msg||''));}
async function door(act){const j=await post('/api/door',{state:act}); msg('m_door', j.ok?'OK':'ERR: '+(j.msg||''));}
async function fan(state){const j=await post('/api/fan',{state}); msg('m_fan', j.ok?'OK':'ERR: '+(j.msg||''));}
async function poll(){
  try{
    const st=await (await fetch('/status',{cache:'no-store'})).json();
    const el=document.getElementById('esp');
    el.className='badge '+(st.esp_connected?'ok':'warn');
    el.textContent=st.esp_connected?'Đã kết nối':'Đang chờ kết nối';
  }catch(e){}
  try{
    const t=await (await fetch('/telemetry',{cache:'no-store'})).json();
    if('temp'in t) document.getElementById('t_temp').textContent=t.temp;
    if('hum' in t) document.getElementById('t_hum').textContent =t.hum;
  }catch(e){}
  setTimeout(poll,2000);
}
poll();
</script>
</html>
"""

@app.route('/')
def index():
    resp = make_response(HTML_TEMPLATE)
    resp.headers["Cache-Control"] = "no-store"
    return resp

# ================= Main =================
def main():
    global _jpeg_quality, _target

    parser = argparse.ArgumentParser()
    parser.add_argument('--esp32_ip', default='auto', help="ESP32 IP address, hoặc 'auto' để tự dò")
    parser.add_argument('--esp32_port', type=int, default=8088)
    parser.add_argument('--camera', default=0, help='Camera index hoặc đường dẫn video (mặc định 0)')
    parser.add_argument('--jpeg-quality', type=int, default=80)
    parser.add_argument('--no-camera', action='store_true', help='Tắt camera/nhận diện, chỉ chạy web + ESP')
    parser.add_argument('--no-gui', action='store_true', help='Không mở cửa sổ OpenCV; vẫn stream qua web')
    args = parser.parse_args()
    _jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))

    # ==== Model paths ====
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'Models')
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'facemodel.pkl')
    FACENET_MODEL_PATH = os.path.join(MODELS_DIR, '20180402-114759.pb')
    ALIGN_DIR = os.path.join(BASE_DIR, 'src', 'align')
    # Nếu không dùng camera, bỏ qua bước check model để dễ chạy nhanh
    if not args.no_camera:
        for f in ("det1.npy", "det2.npy", "det3.npy"):
            if not os.path.exists(os.path.join(ALIGN_DIR, f)):
                raise FileNotFoundError(f"Thiếu {f} tại {ALIGN_DIR}. Hãy đặt 3 file det*.npy vào đây.")

    # ==== Start web server ====
    threading.Thread(target=start_flask_server, daemon=True).start()

    # ==== Auto-discovery / threads ====
    if args.esp32_ip == "auto":
        ip_found = discover_esp_ip(port=args.esp32_port)
        if ip_found:
            _target = {"ip": ip_found, "port": args.esp32_port}
            print(f"[BOOT] ESP32 phát hiện tại {ip_found}:{args.esp32_port}")
        else:
            print("[BOOT] Chưa tìm thấy ESP; sẽ tiếp tục quét nền.")
            _target = {"ip": None, "port": args.esp32_port}
    else:
        _target = {"ip": args.esp32_ip, "port": args.esp32_port}

    _set_esp_state(None, False, _target["ip"], _target["port"], "Chưa kết nối")
    threading.Thread(target=rediscover_loop, daemon=True).start()
    threading.Thread(target=esp_reconnector, args=(_target,), daemon=True).start()
    threading.Thread(target=telemetry_reader, daemon=True).start()

    # ==== Nếu tắt camera, chỉ giữ web + ESP ====
    if args.no_camera:
        print("[INFO] --no-camera: không chạy nhận diện/stream từ camera (trang web vẫn hoạt động).")
        while True:
            time.sleep(1)
        # never returns

    # ==== Load classifier ====
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Không thấy model classifier: {CLASSIFIER_PATH}")
    with open(CLASSIFIER_PATH, 'rb') as f:
        model, class_names = pickle.load(f)
    print("Custom Classifier, Successfully loaded")

    # ==== Facenet graph + camera loop ====
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            if not os.path.exists(FACENET_MODEL_PATH):
                raise FileNotFoundError(f"Không thấy Facenet model: {FACENET_MODEL_PATH}")
            print("Loading feature extraction model")
            facenet.load_model(FACENET_MODEL_PATH)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            pnet, rnet, onet = detect_face.create_mtcnn(sess, ALIGN_DIR)

            cap = VideoStream(src=args.camera).start()
            start_time = 0.0
            sent = False

            try:
                while True:
                    frame = cap.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    frame = imutils.resize(frame, width=640)
                    frame = cv2.flip(frame, 1)

                    boxes, _ = detect_face.detect_face(frame, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)
                    faces = boxes.shape[0] if boxes is not None else 0

                    try:
                        if faces > 0:
                            det = boxes[:, 0:4].astype(np.int32)
                            for i in range(faces):
                                # bỏ qua khuôn mặt quá nhỏ
                                if (det[i][3]-det[i][1]) / frame.shape[0] > 0.25:
                                    cropped = frame[det[i][1]:det[i][3], det[i][0]:det[i][2], :]
                                    scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                                    scaled = facenet.prewhiten(scaled)
                                    feed = {images_placeholder: scaled.reshape(-1, 160, 160, 3),
                                            phase_train_placeholder: False}
                                    emb = sess.run(embeddings, feed_dict=feed)

                                    preds = model.predict_proba(emb)
                                    idx = np.argmax(preds, axis=1)
                                    prob = float(preds[np.arange(len(idx)), idx][0])
                                    name = class_names[idx[0]]

                                    # vẽ nhãn
                                    cv2.rectangle(frame, (det[i][0], det[i][1]), (det[i][2], det[i][3]), (0, 255, 0), 2)
                                    cv2.putText(frame, f"{name} {prob:.2f}",
                                                (det[i][0], max(det[i][1]-10, 20)),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 1)

                                    # Gửi "start" một lần nếu nhận diện đủ tin cậy liên tục >=2s
                                    if prob > 0.8:
                                        if start_time == 0.0:
                                            start_time = time.time()
                                        elif (time.time() - start_time) >= 2.0 and not sent:
                                            ok, msg = send_command("start")
                                            if ok:
                                                print("Đã gửi 'start' tới ESP32")
                                            else:
                                                print("Không gửi được 'start':", msg)
                                            sent = True
                                    else:
                                        start_time = 0.0
                                        sent = False
                        else:
                            start_time = 0.0
                            sent = False
                    except Exception:
                        # giữ ổn định vòng lặp dù lỗi nhỏ
                        pass

                    update_stream_frame(frame)

                    if not args.no_gui:
                        cv2.imshow('Face Recognition', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            finally:
                cap.stop()
                if not args.no_gui:
                    cv2.destroyAllWindows()
                with _esp_lock:
                    if _esp["sock"]:
                        try:
                            _esp["sock"].close()
                        except:
                            pass

if __name__ == "__main__":
    main()
