"""
VozElevate — Ascensor Inteligente Multimodal
Un solo archivo: app.py
Ejecutar con: streamlit run app.py

CORRECCIÓN: eliminado bokeh (incompatible con NumPy >= 1.24).
La voz usa un componente HTML5 con st.components.v1.html
que llama webkitSpeechRecognition y devuelve el texto via URL param.
"""


import re
import time
import json
import pickle
import pathlib
import threading
import numpy as np
import streamlit as st
import html as html_lib 
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ── MQTT ─────────────────────────────────────────────────────
try:
    import paho.mqtt.client as paho
    MQTT_OK = True
except ImportError:
    MQTT_OK = False

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="VozElevate", page_icon="🛗", layout="centered")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:      #0b0f1a;
    --surface: #111827;
    --sur2:    #1a2235;
    --border:  #1e2d45;
    --cyan:    #00e5ff;
    --indigo:  #6366f1;
    --green:   #00e676;
    --amber:   #ffab00;
    --red:     #ff1744;
    --text:    #e2e8f0;
    --muted:   #64748b;
}
html, body, .stApp { background-color: var(--bg) !important; }
* { font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 480px !important; padding: 1.5rem 1rem 5rem !important; margin: 0 auto !important; }

.topbar { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0.2rem 1.4rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem; }
.topbar-logo { font-size:1.1rem; font-weight:800; letter-spacing:0.06em; color:var(--cyan); text-shadow:0 0 12px rgba(0,229,255,0.5); }
.topbar-sub  { font-size:0.65rem; font-weight:600; color:var(--muted); letter-spacing:0.12em; text-transform:uppercase; }

.elev-card { background:var(--surface); border:1px solid var(--border); border-radius:24px; padding:1.6rem 1.4rem 1.4rem; margin-bottom:1.2rem; position:relative; overflow:hidden; box-shadow:0 0 24px rgba(0,229,255,0.1); }
.elev-card::before { content:''; position:absolute; top:0; left:50%; transform:translateX(-50%); width:60%; height:1px; background:linear-gradient(90deg,transparent,var(--cyan),transparent); }
.floor-label { font-size:0.65rem; font-weight:600; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; text-align:center; margin-bottom:0.2rem; }
.floor-num   { font-family:'JetBrains Mono',monospace !important; font-size:4.5rem; font-weight:600; color:var(--cyan); text-shadow:0 0 32px rgba(0,229,255,0.45); line-height:1; text-align:center; }
.status-bar  { display:flex; align-items:center; justify-content:center; background:var(--sur2); border:1px solid var(--border); border-radius:12px; padding:0.55rem 1rem; font-size:0.85rem; font-weight:600; color:var(--text); min-height:2.4rem; margin-top:0.8rem; }

.leds { display:flex; justify-content:center; gap:0.6rem; margin-top:0.9rem; }
.led  { width:10px; height:10px; border-radius:50%; background:var(--border); }
.led.green  { background:var(--green); box-shadow:0 0 8px var(--green); }
.led.amber  { background:var(--amber); box-shadow:0 0 8px var(--amber); animation:pulse 0.8s infinite; }
.led.red    { background:var(--red);   box-shadow:0 0 8px var(--red);   animation:pulse 0.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.sec-label { font-size:0.62rem; font-weight:700; color:var(--muted); letter-spacing:0.14em; text-transform:uppercase; margin:1.2rem 0 0.55rem 0.1rem; }

.info-card { background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:1rem 1.2rem; margin-bottom:0.7rem; }
.info-row  { display:flex; align-items:center; gap:0.7rem; }
.info-icon { font-size:1.4rem; flex-shrink:0; }
.info-ttl  { font-size:0.68rem; font-weight:600; color:var(--muted); margin-bottom:0.1rem; }
.info-val  { font-size:0.92rem; font-weight:700; color:var(--text); }
.info-sub  { font-size:0.76rem; color:var(--muted); margin-top:0.1rem; }

.result-pill { background:var(--sur2); border:1px solid var(--border); border-radius:14px; padding:0.8rem 1rem; font-size:0.88rem; font-weight:600; color:var(--text); margin-top:0.6rem; display:flex; align-items:flex-start; gap:0.5rem; }
.mqtt-badge { font-size:0.7rem; font-weight:600; color:var(--muted); text-align:center; padding:0.4rem; background:var(--surface); border-radius:8px; border:1px solid var(--border); margin-bottom:0.5rem; }
.hist-item  { font-family:'JetBrains Mono',monospace !important; font-size:0.72rem; color:var(--muted); padding:0.3rem 0; border-bottom:1px solid var(--border); }

div[data-testid="stButton"]>button {
    background:linear-gradient(135deg,var(--indigo),#4f46e5) !important;
    color:white !important; border:none !important; border-radius:14px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    width:100% !important; transition:all 0.2s !important;
    box-shadow:0 4px 14px rgba(99,102,241,0.3) !important;
}
div[data-testid="stButton"]>button:hover { transform:translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MQTT / WOKWI
# ─────────────────────────────────────────────────────────────
BROKER = "broker.mqttdashboard.com"
PORT   = 1883
TOPIC  = "vozelevate/cmd"

class WokwiBridge:
    def __init__(self):
        self._client = None
        self._connected = False
        self._error = None
        if MQTT_OK:
            try:
                self._client = paho.Client("VozElevate_App")
                self._client.on_connect    = lambda c,u,f,rc: setattr(self,'_connected', rc==0)
                self._client.on_disconnect = lambda c,u,rc:   setattr(self,'_connected', False)
                threading.Thread(target=self._connect, daemon=True).start()
            except Exception as e:
                self._error = str(e)

    def _connect(self):
        try:
            self._client.connect(BROKER, PORT, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            self._error = str(e)

    def publish(self, payload):
        if self._client:
            try: self._client.publish(TOPIC, json.dumps(payload))
            except: pass

    def send_move(self, floor):    self.publish({"cmd":"MOVE",    "floor":floor})
    def send_arrived(self, floor): self.publish({"cmd":"ARRIVED", "floor":floor})
    def send_emergency(self):      self.publish({"cmd":"EMERGENCY"})
    def send_reset(self):          self.publish({"cmd":"RESET"})

    def status(self):
        if not MQTT_OK:      return "⚠️ paho-mqtt no instalado"
        if self._connected:  return f"✅ Conectado · {BROKER}"
        if self._error:      return f"❌ {self._error}"
        return "🔄 Conectando…"

# ─────────────────────────────────────────────────────────────
# LÓGICA DEL ASCENSOR
# ─────────────────────────────────────────────────────────────
def init_elevator():
    return {"current":1, "target":None, "state":"idle",
            "message":"Selecciona un piso", "history":[]}

def go_to_floor(ev, floor):
    if floor == ev["current"]:
        ev["message"] = f"Ya estás en el piso {floor} 👌"
        ev["state"]   = "idle"
        return
    ev["target"]  = floor
    ev["state"]   = "moving"
    direction     = "⬆ Subiendo" if floor > ev["current"] else "⬇ Bajando"
    ev["message"] = f"{direction} al piso {floor}…"
    ev["history"].append(f"[{time.strftime('%H:%M:%S')}] Piso {ev['current']} → {floor}")

def arrive(ev):
    ev["current"] = ev["target"]
    ev["target"]  = None
    ev["state"]   = "arrived"
    ev["message"] = f"🎉 Llegaste al piso {ev['current']}. Puertas abiertas."

def set_emergency(ev):
    ev["target"]  = None
    ev["state"]   = "emergency"
    ev["message"] = "🚨 Emergencia activada."
    ev["history"].append(f"[{time.strftime('%H:%M:%S')}] EMERGENCIA en piso {ev['current']}")

def reset_ev(ev):
    ev["state"]   = "idle"
    ev["message"] = "Selecciona un piso"

# ─────────────────────────────────────────────────────────────
# PARSER DE VOZ
# ─────────────────────────────────────────────────────────────
WORD2NUM = {
    "uno":1,"primero":1,"dos":2,"segundo":2,"tres":3,"tercero":3,
    "cuatro":4,"cuarto":4,"cinco":5,"quinto":5,"seis":6,"sexto":6,
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
}
EMERGENCY_WORDS = {"emergencia","emergency","cancelar","cancel","paro","stop"}

def parse_voice(text):
    t = text.lower().strip()
    for w in EMERGENCY_WORDS:
        if w in t: return {"emergency": True}
    nums = re.findall(r"\b([1-6])\b", t)
    if nums: return {"floor": int(nums[0])}
    for word, num in WORD2NUM.items():
        if word in t: return {"floor": num}
    return {"error": f"No entendí el piso en: \"{text}\""}

# ─────────────────────────────────────────────────────────────
# RECONOCIMIENTO DE DÍGITOS
# ─────────────────────────────────────────────────────────────
def predict_digit(pil_image):
    try:
        model = tf.keras.models.load_model("model/handwritten.h5")
        image = ImageOps.grayscale(pil_image)
        img   = image.resize((28, 28))
        img   = np.array(img, dtype="float32") / 255.0
        img   = img.reshape((1, 28, 28, 1))
        pred  = model.predict(img)
        return int(np.argmax(pred[0]))
    except Exception as e:
        return None
        
# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "ev"         not in st.session_state: st.session_state.ev         = init_elevator()
if "bridge"     not in st.session_state: st.session_state.bridge     = WokwiBridge()
if "feedback"   not in st.session_state: st.session_state.feedback   = None
if "draw_digit" not in st.session_state: st.session_state.draw_digit = None
if "voice_text" not in st.session_state: st.session_state.voice_text = ""

ev = st.session_state.ev
br = st.session_state.bridge

# ─────────────────────────────────────────────────────────────
# ACCIÓN CENTRAL
# ─────────────────────────────────────────────────────────────
def action_go(floor, source):
    if ev["state"] == "emergency":
        st.session_state.feedback = ("red", "🚨 Emergencia activa. Cancela primero.")
        return
    go_to_floor(ev, floor)
    if ev["state"] == "moving":
        br.send_move(floor)
        st.session_state.feedback = ("amber", f"[{source}] {ev['message']}")
        steps = abs(floor - ev["current"]) if ev["target"] is None else abs(floor - ev["current"])
        steps = max(1, steps)
        bar = st.progress(0, text=ev["message"])
        for i in range(steps):
            time.sleep(0.7)
            bar.progress((i + 1) / steps, text=ev["message"])
        bar.empty()
        arrive(ev)
        br.send_arrived(ev["current"])
        st.session_state.feedback = ("green", ev["message"])
        reset_ev(ev)
    else:
        st.session_state.feedback = ("green", ev["message"])

# ─────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-logo">🛗 VozElevate</div>
        <div class="topbar-sub">Ascensor Multimodal</div>
    </div>
    <div style="font-size:1.2rem;display:flex;gap:0.5rem">⚙️ 🔔</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TARJETA DEL ASCENSOR
# ─────────────────────────────────────────────────────────────
 
state = ev["state"]
led_g = "green" if state in ("idle", "arrived") else ""
led_a = "amber" if state == "moving"            else ""
led_r = "red"   if state == "emergency"         else ""
 
# Escapar caracteres especiales para que no rompan el HTML
current = ev["current"]
msg     = html_lib.escape(str(ev["message"]))
 
dest_html = ""
if ev["target"]:
    dest_html = (
        '<div style="text-align:center;margin-top:0.5rem">'
        '<span style="font-size:0.65rem;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.1em">DESTINO</span><br>'
        '<span style="font-family:JetBrains Mono,monospace;font-size:1.8rem;'
        'color:#6366f1;font-weight:600">' + str(ev["target"]) + '</span>'
        '</div>'
    )
 
card_html = (
    '<div class="elev-card">'
    '<div class="floor-label">PISO ACTUAL</div>'
    '<div class="floor-num">' + str(current) + '</div>'
    + dest_html +
    '<div class="status-bar">' + msg + '</div>'
    '<div class="leds">'
    '<div class="led ' + led_g + '"></div>'
    '<div class="led ' + led_a + '"></div>'
    '<div class="led ' + led_r + '"></div>'
    '</div>'
    '</div>'
)
 
st.markdown(card_html, unsafe_allow_html=True)
# Feedback
fb = st.session_state.feedback
if fb:
    color, msg = fb
    icon = {"green":"✔","amber":"⬆","red":"🚨"}.get(color,"•")
    st.markdown(f"""
    <div class="result-pill">
        <span>{icon}</span>
        <div>
            <div style="color:#64748b;font-size:0.68rem;font-weight:600;margin-bottom:0.1rem">ESTADO</div>
            {msg}
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f'<div class="mqtt-badge">Wokwi MQTT · {br.status()}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_voz, tab_btn, tab_draw, tab_hist = st.tabs(["🎤 Voz", "🔢 Botones", "✏️ Dibujo", "📋 Historial"])

# ══ TAB VOZ ══════════════════════════════════════════════════
with tab_voz:
    st.markdown("""
    <div class="info-card">
        <div class="info-row">
            <span class="info-icon">🎤</span>
            <div>
                <div class="info-ttl">INSTRUCCIÓN</div>
                <div class="info-val">Presiona el micrófono y habla</div>
                <div class="info-sub">Ej: "Ir al piso cuatro" · "Tres" · "Emergencia"</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Componente HTML5 con webkitSpeechRecognition
    # Escribe el resultado en un input oculto que Streamlit lee via query_params
    voice_html = """
    <style>
        #mic-btn {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white; border: none; border-radius: 50px;
            padding: 0.6rem 2rem; font-size: 1rem; font-weight: 700;
            cursor: pointer; width: 100%; font-family: 'Syne', sans-serif;
            box-shadow: 0 4px 14px rgba(99,102,241,0.4);
            transition: all 0.2s;
        }
        #mic-btn:hover { transform: translateY(-1px); }
        #mic-btn.listening {
            background: linear-gradient(135deg, #ff1744, #b71c1c);
            animation: pulse-btn 0.8s infinite;
        }
        @keyframes pulse-btn { 0%,100%{opacity:1} 50%{opacity:0.7} }
        #voice-out {
            margin-top: 0.7rem; padding: 0.6rem 0.9rem;
            background: #1a2235; border: 1px solid #1e2d45;
            border-radius: 10px; color: #e2e8f0;
            font-family: 'Syne', sans-serif; font-size: 0.88rem;
            min-height: 2rem;
        }
    </style>

    <button id="mic-btn" onclick="startListen()">🎙 Iniciar escucha</button>
    <div id="voice-out">Toca el botón y habla…</div>
    <input type="hidden" id="voice-result" value="">

    <script>
    var recognizing = false;
    var recognition;

    function startListen() {
        if (!('webkitSpeechRecognition' in window)) {
            document.getElementById('voice-out').innerText = '⚠️ Tu navegador no soporta reconocimiento de voz. Usa Chrome.';
            return;
        }
        if (recognizing) { recognition.stop(); return; }

        recognition = new webkitSpeechRecognition();
        recognition.lang = 'es-ES';
        recognition.continuous = false;
        recognition.interimResults = false;

        var btn = document.getElementById('mic-btn');
        btn.classList.add('listening');
        btn.innerText = '⏹ Escuchando… (toca para parar)';
        recognizing = true;

        recognition.onresult = function(e) {
            var texto = e.results[0][0].transcript;
            document.getElementById('voice-out').innerText = '🗣️ Escuché: "' + texto + '"';
            document.getElementById('voice-result').value = texto;
            // Envía el texto a Streamlit via postMessage
            window.parent.postMessage({type: 'VOICE_RESULT', text: texto}, '*');
        };

        recognition.onerror = function(e) {
            document.getElementById('voice-out').innerText = '⚠️ Error: ' + e.error;
        };

        recognition.onend = function() {
            recognizing = false;
            btn.classList.remove('listening');
            btn.innerText = '🎙 Iniciar escucha';
        };

        recognition.start();
    }
    </script>
    """

    # Renderizamos el componente de voz
    components.html(voice_html, height=130)

    st.markdown("---")
    st.markdown('<div class="sec-label">✍️ O escribe el comando manualmente</div>', unsafe_allow_html=True)

    col_inp, col_btn = st.columns([3, 1])
    with col_inp:
        voz_txt = st.text_input(
            "Comando de voz",
            placeholder="Ej: ir al piso 4 · tres · emergencia",
            key="voz_input",
            label_visibility="collapsed"
        )
    with col_btn:
        if st.button("Enviar", key="voz_send"):
            if voz_txt.strip():
                parsed = parse_voice(voz_txt)
                if "emergency" in parsed:
                    set_emergency(ev); br.send_emergency()
                    st.session_state.feedback = ("red", ev["message"])
                elif "floor" in parsed:
                    action_go(parsed["floor"], "voz")
                else:
                    st.session_state.feedback = ("red", parsed.get("error",""))
                st.rerun()

    st.caption("💡 El micrófono funciona en Chrome/Edge. El resultado de voz también puedes escribirlo arriba.")

# ══ TAB BOTONES ══════════════════════════════════════════════
with tab_btn:
    st.markdown('<div class="sec-label">🔢 Selecciona el piso</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, floor in enumerate([1, 2, 3, 4, 5, 6]):
        with cols[i % 3]:
            if st.button(f"  {floor}  ", key=f"floor_{floor}"):
                action_go(floor, "botón")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚨 Emergencia", key="emg"):
            set_emergency(ev); br.send_emergency()
            st.session_state.feedback = ("red", ev["message"])
            st.rerun()
    with c2:
        if st.button("↺ Reiniciar", key="rst"):
            reset_ev(ev); br.send_reset()
            st.session_state.feedback = ("green", "Sistema reiniciado.")
            st.rerun()

# ══ TAB DIBUJO ═══════════════════════════════════════════════
with tab_draw:
    st.markdown("""
    <div class="info-card">
        <div class="info-row">
            <span class="info-icon">✏️</span>
            <div>
                <div class="info-ttl">INSTRUCCIÓN</div>
                <div class="info-val">Dibuja el número del piso</div>
                <div class="info-sub">Escribe un número del 1 al 6 en el lienzo</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stroke_width = st.slider("Grosor del trazo", 8, 30, 16, key="stroke")


    st.markdown("""
    <style>
    canvas {
    border: 2px solid #00e5ff !important;
    border-radius: 12px !important;
    display: block !important;
    visibility: visible !important;
    }
    </style>
    
""", unsafe_allow_html=True)

    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color="#00e5ff",      # trazo cyan como tu tema
    background_color="#1a2235",  # ← oscuro pero visible contra tu app
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Reconocer dígito", key="predict"):
            if canvas_result.image_data is not None:
                input_image = Image.fromarray(
                    np.array(canvas_result.image_data).astype("uint8"), "RGBA"
                )
                digit = predict_digit(input_image)
                st.session_state.draw_digit = digit
                if digit is None:
                    st.session_state.feedback = ("red", "No se pudo reconocer. Intenta de nuevo.")
                elif not (1 <= digit <= 6):
                    st.session_state.feedback = ("amber", f"Dígito {digit} fuera de rango (1-6).")
                else:
                    action_go(digit, "dibujo")
                st.rerun()
            else:
                st.session_state.feedback = ("red", "El lienzo está vacío.")
                st.rerun()
    with c2:
        if st.button("🗑 Limpiar", key="clear_canvas"):
            st.session_state.draw_digit = None
            st.rerun()

    if st.session_state.draw_digit is not None:
        st.markdown(f"""
        <div class="result-pill">
            <span>🔢</span>
            <div>
                <div style="color:#64748b;font-size:0.68rem;font-weight:600">DÍGITO DETECTADO</div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:2rem;color:#00e5ff">{st.session_state.draw_digit}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        
# ══ TAB HISTORIAL ════════════════════════════════════════════
with tab_hist:
    if not ev["history"]:
        st.markdown('<div style="color:#64748b;font-size:0.85rem;text-align:center;padding:1.5rem 0">Sin viajes registrados aún</div>', unsafe_allow_html=True)
    else:
        for item in reversed(ev["history"][-20:]):
            st.markdown(f'<div class="hist-item">› {item}</div>', unsafe_allow_html=True)
        if st.button("🗑 Limpiar historial", key="clear_hist"):
            ev["history"].clear()
            st.rerun()
