"""
Face recognition + ESP32 TCP control + Flask MJPEG web + robust auto-discovery.

Endpoints:
- /               : UI xem video + LED + Cửa (Servo) + Telemetry
- /video_feed     : MJPEG stream
- /status         : trạng thái kết nối ESP
- /telemetry      : JSON cảm biến/ACK mới nhất
- /logo           : trả về /src/images/Logo.png

Auto-discovery:
- Nếu --esp32_ip=auto: quét subnet /24, ưu tiên ARP; chỉ nhận thiết bị gửi '{"hello":"esp32"}'.
- Không hiển thị nút START; nhưng tự động gửi 'start' khi vừa kết nối thành công.
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

# TensorFlow v1 style (dùng facenet + MTCNN)
import tensorflow as tf
import facenet
from align import detect_face as detect_face
from sklearn.svm import SVC

from flask import Flask, Response, make_response, jsonify, request, send_file
from flask_cors import CORS

# ---- Cờ tự động gửi 'start' khi kết nối ESP thành công
AUTO_START_ON_CONNECT = True

# -------------------- Flask & stream --------------------
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
            time.sleep(0.02)
            continue
        yield boundary + b'\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- ESP state & telemetry --------------------
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

# ---- Logo route ----
@app.route('/logo')
def logo():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, 'src', 'images', 'Logo.png')
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    return Response(status=404)

def _set_esp_state(sock, connected, ip, port, last_error=""):
    with _esp_lock:
        _esp.update({"sock": sock, "connected": connected, "ip": ip or "", "port": port, "last_error": last_error})

# -------------------- GỬI LỆNH (tập trung) --------------------
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

def send_command(kind: str, value=None):
    """
    Hỗ trợ:
      - led:on/off
      - servo:<angle>
      - door:open / door:close (gửi qua servo 0/180)
      - start (giữ tương thích)
      - ping
      - raw:<string>
    """
    kind = (kind or "").lower().strip()

    if kind == "led":
        state = str(value).lower()
        if state not in ("on", "off"): return False, "invalid led state"
        return send_to_esp_raw(f"led:{state}")

    if kind == "servo":
        try:
            angle = int(value)
        except Exception:
            return False, "invalid servo angle"
        angle = max(0, min(180, angle))
        return send_to_esp_raw(f"servo:{angle}")

    if kind == "door":
        state = str(value).lower()
        if state == "open":  return send_to_esp_raw("door:open")
        if state == "close": return send_to_esp_raw("door:close")
        return False, "invalid door state"

    if kind == "start":
        return send_to_esp_raw("start")

    if kind == "ping":
        return send_to_esp_raw("ping")

    if kind == "raw":
        if not isinstance(value, str) or not value:
            return False, "invalid raw command"
        return send_to_esp_raw(value)

    return False, "unknown command"

# -------------------- API (chỉ LED & SERVO/DOOR) --------------------
@app.route('/api/led', methods=['POST'])
def api_led():
    state = str((request.get_json(silent=True) or {}).get("state", "on")).lower()
    if state not in ("on", "off"):
        return ({"ok": False, "msg": "state must be on/off"}, 400)
    ok, msg = send_command("led", state)
    return ({"ok": ok, "msg": msg, "state": state}, 200 if ok else 503)

@app.route('/api/servo', methods=['POST'])
def api_servo():
    data = request.get_json(silent=True) or {}
    angle = data.get("angle", 90)
    try:
        angle = int(angle)
    except Exception:
        angle = 90
    angle = max(0, min(180, angle))
    ok, msg = send_command("servo", angle)
    return ({"ok": ok, "msg": msg, "angle": angle}, 200 if ok else 503)

# -------------------- Helper: LAN IP & Flask start --------------------
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

# -------------------- Discovery (xác thực hello) --------------------
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

def _probe_esp(ip, port, timeout=0.4):
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
        try: s.close()
        except: pass
        return False

def discover_esp_ip(port=8088, timeout_per_host=0.4, max_workers=64):
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

# -------------------- Kết nối ESP --------------------
_target_lock = threading.Lock()
_target = {"ip": None, "port": 8088}

def esp_reconnector(target_ref, interval=8):
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

                # Auto-START ngay sau khi kết nối
                if AUTO_START_ON_CONNECT:
                    ok2, msg2 = send_command("start")
                    if ok2:
                        print("[ESP] Auto-START đã gửi sau khi kết nối.")
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

# -------------------- UI (nền trắng, logo, không có nút START) --------------------
HTML_TEMPLATE = """
<!doctype html><html lang="vi">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>FaceCam</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#ffffff;color:#111;font-family:system-ui,Segoe UI,Roboto;line-height:1.45}
.wrap{min-height:100vh;display:flex;flex-direction:column;gap:16px;align-items:center;justify-content:flex-start;padding:16px}
.header{width:min(100%,1000px);display:flex;align-items:center;gap:12px}
.header img{height:46px;width:auto}
.status{margin-left:auto;padding:10px 14px;border-radius:10px;border:1px solid #e5e7eb;background:#fafafa;font-size:clamp(12px,2.8vw,14px);word-break:break-word}
.ok{background:#e8f7ef;border-color:#b7e3c7;color:#066a36}
.warn{background:#fff3f3;border-color:#ffd6d6;color:#991b1b}
.media{width:min(100%,1000px)}
.media img{width:100%;height:auto;max-height:66vh;object-fit:contain;border-radius:12px;border:1px solid #e5e7eb;background:#fff}
.panel{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;width:min(100%,1000px)}
.card{border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;background:#ffffff}
.card h3{margin:0 0 8px 0;font-size:clamp(15px,3.5vw,16px);color:#111}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,input{padding:10px 12px;border-radius:10px;border:1px solid #d1d5db;background:#f3f4f6;color:#111;font-size:clamp(13px,3.5vw,14px)}
button:active{transform:scale(0.98)}
.kv{display:grid;grid-template-columns:120px 1fr;gap:6px 10px;font-size:clamp(12px,3.2vw,14px)}
.value{font-weight:700}.hint{opacity:.8;font-size:clamp(12px,3.2vw,13px)}
.note{width:min(100%,1000px)}
code{background:#f3f4f6;padding:2px 6px;border-radius:6px;word-break:break-all;border:1px solid #e5e7eb}
@media (max-width:600px){
  .kv{grid-template-columns:100px 1fr}
  .row > *{flex:1}
  button{width:48%}
  .header{flex-direction:column;align-items:flex-start;gap:8px}
  .status{margin-left:0;width:100%}
}
</style>
<div class="wrap">
  <div class="header">
    <img src="/logo" alt="Logo" onerror="this.style.display='none'">
    <div class="status warn" id="esp">Đang kiểm tra kết nối ESP32...</div>
  </div>

  <div class="media"><img src="/video_feed" alt="Live video"></div>

  <div class="panel">
    <div class="card"><h3>Đèn LED</h3>
      <div class="row">
        <button onclick="led('on')">Bật LED</button>
        <button onclick="led('off')">Tắt LED</button>
      </div>
      <div id="led_msg" class="hint"></div>
    </div>

    <div class="card"><h3>Cửa (Servo)</h3>
      <div class="row">
        <button onclick="servo_move(0)">Đóng cửa (0°)</button>
        <button onclick="servo_move(180)">Mở cửa (180°)</button>
      </div>
      <div id="servo_msg" class="hint"></div>
    </div>

    <div class="card"><h3>Cảm biến</h3>
      <div class="kv">
        <div>Nhiệt độ</div><div><span id="temp" class="value">—</span> °C</div>
        <div>Độ ẩm</div><div><span id="hum" class="value">—</span> %</div>
        <div>Gas (raw)</div><div><span id="gas_raw" class="value">—</span></div>
        <div>Gas (V)</div><div><span id="gas_v" class="value">—</span> V</div>
        <div>RSSI</div><div><span id="rssi" class="value">—</span> dBm</div>
        <div>Uptime</div><div><span id="uptime" class="value">—</span> s</div>
      </div>
    </div>
  </div>

  <div class="note hint">Truy cập từ thiết bị khác trong cùng Wi-Fi: <code>http://&lt;IP-LAN-PC&gt;:5000/</code></div>
</div>

<script>
async function led(state){
  const r=await fetch('/api/led',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state})});
  const j=await r.json(); led_msg.textContent=j.ok?'OK':'ERR: '+j.msg;
}
async function servo_move(deg){
  const r=await fetch('/api/servo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({angle:deg})});
  const j=await r.json(); servo_msg.textContent=j.ok?'OK':'ERR: '+j.msg;
}
async function poll(){
  try{
    const r=await fetch('/status',{cache:'no-store'}); const st=await r.json();
    const el=document.getElementById('esp');
    el.className='status '+(st.esp_connected?'ok':'warn');
    el.textContent=st.esp_connected?('ESP32 đã kết nối: '+st.ip+':'+st.port)
      :('Đang chờ kết nối ESP32'
        +(st.ip?' ('+st.ip+':'+st.port+')':'')
        +(st.last_error?' — '+st.last_error:'')+'...');
  }catch(e){}
  try{
    const t=await fetch('/telemetry',{cache:'no-store'});
    if(t.ok){
      const d=await t.json();
      if('temp' in d)   temp.textContent=d.temp;
      if('hum' in d)    hum.textContent =d.hum;
      if('gas_raw' in d) gas_raw.textContent=d.gas_raw;
      if('gas_v' in d)  gas_v.textContent=d.gas_v;
      if('rssi' in d)   rssi.textContent=d.rssi;
      if('uptime' in d) uptime.textContent=d.uptime;
    }
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

# -------------------- Main --------------------
def main():
    global _jpeg_quality, _target

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=0, help='Camera index or video path. Default camera 0.')
    parser.add_argument('--esp32_ip', default='auto', help="ESP32 IP address, hoặc 'auto' để tự dò")
    parser.add_argument('--esp32_port', type=int, default=8088)
    parser.add_argument('--no-gui', action='store_true', help='Tắt cửa sổ OpenCV, chỉ dùng web')
    parser.add_argument('--jpeg-quality', type=int, default=80)
    args = parser.parse_args()
    _jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))

    # ==== Model paths ====
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'Models')
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'facemodel.pkl')
    FACENET_MODEL_PATH = os.path.join(MODELS_DIR, '20180402-114759.pb')
    ALIGN_DIR = os.path.join(BASE_DIR, 'src', 'align')
    for f in ("det1.npy", "det2.npy", "det3.npy"):
        if not os.path.exists(os.path.join(ALIGN_DIR, f)):
            raise FileNotFoundError(f"Thiếu {f} tại {ALIGN_DIR}. Hãy đặt 3 file det*.npy vào đây.")

    # ==== Start web server ====
    threading.Thread(target=start_flask_server, daemon=True).start()

    # ==== Auto-discovery / set target / threads ====
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

            cap = VideoStream(src=args.path).start()
            start_time = 0.0
            sent = False

            try:
                while True:
                    frame = cap.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    frame = imutils.resize(frame, width=600)
                    frame = cv2.flip(frame, 1)

                    boxes, _ = detect_face.detect_face(
                        frame, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709
                    )
                    faces = boxes.shape[0] if boxes is not None else 0

                    try:
                        if faces > 0:
                            det = boxes[:, 0:4].astype(np.int32)
                            for i in range(faces):
                                if (det[i][3]-det[i][1]) / frame.shape[0] > 0.25:
                                    cropped = frame[det[i][1]:det[i][3], det[i][0]:det[i][2], :]
                                    scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                                    scaled = facenet.prewhiten(scaled)
                                    feed = {images_placeholder: scaled.reshape(-1, 160, 160, 3), phase_train_placeholder: False}
                                    emb = sess.run(embeddings, feed_dict=feed)

                                    preds = model.predict_proba(emb)
                                    idx = np.argmax(preds, axis=1)
                                    prob = float(preds[np.arange(len(idx)), idx][0])
                                    name = class_names[idx[0]]

                                    if prob > 0.8:
                                        cv2.rectangle(frame, (det[i][0], det[i][1]), (det[i][2], det[i][3]), (0, 255, 0), 2)
                                        cv2.putText(frame, f"{name} {prob:.3f}", (det[i][0], max(det[i][1]-10, 20)),
                                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

                                        if start_time == 0.0:
                                            start_time = time.time()
                                        else:
                                            if (time.time() - start_time) >= 2.0 and not sent:
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
