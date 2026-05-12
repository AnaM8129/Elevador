"""
VozElevate — Ascensor Inteligente Multimodal
============================================
Modalidades de entrada:
  · Voz       → mic_recorder + Google Speech Recognition
  · Botones   → Streamlit buttons (sidebar)
  · Dibujo    → Canvas dibujable → CNN MNIST → reconoce número 1-6

Comunicación con Wokwi:
  · MQTT: broker.mqttdashboard.com  topic: vozelevate/cmd
  · Payload JSON: {"piso": 4, "accion": "mover"}
                  {"accion": "emergencia"}
                  {"accion": "cancelar"}
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import io
import time
import json
import re

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Voz
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# Dibujo
from streamlit_drawable_canvas import st_canvas

# MQTT
import paho.mqtt.client as paho

# Modelo CNN (se carga una sola vez con caché)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════
BROKER    = "broker.mqttdashboard.com"
PORT      = 1883
TOPIC     = "vozelevate/cmd"
CLIENT_ID = "VozElevate_App"

PISOS_VALIDOS = list(range(1, 7))
ANGULOS_SERVO = {1: 0, 2: 30, 3: 60, 4: 90, 5: 120, 6: 150}

PALABRAS_NUMERO = {
    "uno": 1, "one": 1, "1": 1,
    "dos": 2, "two": 2, "2": 2,
    "tres": 3, "three": 3, "3": 3,
    "cuatro": 4, "four": 4, "4": 4,
    "cinco": 5, "five": 5, "5": 5,
    "seis": 6, "six": 6, "6": 6,
}

MODEL_PATH = "model/handwritten.h5"


# ══════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "piso_actual":  1,
        "piso_destino": None,
        "estado":       "disponible",   # disponible | moviendo | emergencia
        "log":          [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
# MQTT
# ══════════════════════════════════════════════════════════════
def mqtt_publicar(payload: dict) -> bool:
    try:
        c = paho.Client(CLIENT_ID + "_pub")
        c.connect(BROKER, PORT, keepalive=5)
        result = c.publish(TOPIC, json.dumps(payload))
        c.disconnect()
        return result.rc == paho.MQTT_ERR_SUCCESS
    except Exception as e:
        agregar_log(f"⚠️ MQTT error: {e}", "error")
        return False


# ══════════════════════════════════════════════════════════════
# LÓGICA CENTRAL DEL ASCENSOR
# ══════════════════════════════════════════════════════════════
def agregar_log(msg: str, tipo: str = "info"):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log.append({"t": timestamp, "msg": msg, "tipo": tipo})
    if len(st.session_state.log) > 50:
        st.session_state.log = st.session_state.log[-50:]


def mover_ascensor(piso_destino: int):
    if st.session_state.estado == "emergencia":
        st.warning("⚠️ Hay una emergencia activa. Cancélala primero.")
        return
    if piso_destino == st.session_state.piso_actual:
        agregar_log(f"Ya estás en el piso {piso_destino}.", "info")
        return
    if piso_destino not in PISOS_VALIDOS:
        agregar_log(f"Piso {piso_destino} no válido.", "error")
        return

    st.session_state.piso_destino = piso_destino
    st.session_state.estado = "moviendo"
    direccion = "⬆️ Subiendo" if piso_destino > st.session_state.piso_actual else "⬇️ Bajando"

    agregar_log(f"Destino seleccionado: Piso {piso_destino}", "ok")
    agregar_log(f"{direccion} al piso {piso_destino}...", "info")

    mqtt_publicar({
        "accion": "mover",
        "piso":   piso_destino,
        "desde":  st.session_state.piso_actual,
        "angulo": ANGULOS_SERVO[piso_destino],
    })

    paso = 1 if piso_destino > st.session_state.piso_actual else -1
    piso_tmp = st.session_state.piso_actual
    placeholder = st.empty()

    while piso_tmp != piso_destino:
        piso_tmp += paso
        st.session_state.piso_actual = piso_tmp
        dir_icon = "⬆️" if paso == 1 else "⬇️"
        with placeholder.container():
            st.info(f"{dir_icon} Pasando por piso **{piso_tmp}**...")
        time.sleep(0.9)

    placeholder.empty()

    st.session_state.estado = "disponible"
    st.session_state.piso_destino = None
    agregar_log(f"✅ Llegaste al piso {piso_destino}. Puertas abiertas.", "ok")

    mqtt_publicar({
        "accion": "llegada",
        "piso":   piso_destino,
        "angulo": ANGULOS_SERVO[piso_destino],
    })


def activar_emergencia():
    st.session_state.estado = "emergencia"
    agregar_log("🚨 EMERGENCIA activada. Ascensor detenido.", "error")
    mqtt_publicar({"accion": "emergencia", "piso": st.session_state.piso_actual})


def cancelar_emergencia():
    st.session_state.estado = "disponible"
    agregar_log("✅ Emergencia cancelada. Sistema disponible.", "ok")
    mqtt_publicar({"accion": "cancelar"})


# ══════════════════════════════════════════════════════════════
# RECONOCIMIENTO DE VOZ
# ══════════════════════════════════════════════════════════════
def extraer_piso_de_texto(texto: str):
    texto = texto.lower().strip()
    matches = re.findall(r"\b([1-6])\b", texto)
    if matches:
        return int(matches[0])
    for palabra, num in PALABRAS_NUMERO.items():
        if palabra in texto:
            return num
    return None


def reconocer_audio(audio_bytes: bytes):
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language="es-ES")
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        agregar_log(f"Error Google Speech: {e}", "error")
        return None
    except Exception as e:
        agregar_log(f"Error audio: {e}", "error")
        return None


# ══════════════════════════════════════════════════════════════
# RECONOCIMIENTO DE DIBUJO
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def predecir_digito(imagen_rgba: np.ndarray):
    modelo = cargar_modelo()
    if modelo is None:
        return None
    try:
        img = Image.fromarray(imagen_rgba.astype("uint8"), "RGBA")
        img = img.convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        arr = np.array(img, dtype="float32") / 255.0
        arr = arr.reshape((1, 28, 28, 1))
        pred = modelo.predict(arr, verbose=0)
        return int(np.argmax(pred[0]))
    except Exception as e:
        agregar_log(f"CNN error: {e}", "error")
        return None


# ══════════════════════════════════════════════════════════════
# ESTILOS CSS
# ══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');

:root {
    --bg:     #0A0F1E;
    --card:   #111827;
    --panel:  #1A2235;
    --cyan:   #00E5FF;
    --teal:   #00B4CC;
    --aqua:   #00FFD1;
    --white:  #FFFFFF;
    --silver: #CBD5E1;
    --fog:    #64748B;
    --green:  #00FF94;
    --amber:  #FFB800;
    --red:    #FF3B5C;
    --lilac:  #C084FC;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Quicksand', sans-serif !important;
    color: var(--white) !important;
}
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] * { color: var(--silver) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 16px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--fog) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--cyan) !important;
    color: var(--bg) !important;
}

.hud-bar {
    background: var(--card);
    border: 1px solid #1E2D45;
    border-radius: 20px;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
}
.hud-piso {
    font-size: 3rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
}
.hud-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--fog);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.estado-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.estado-disponible { background: rgba(0,255,148,0.15); color: var(--green); border: 1px solid rgba(0,255,148,0.3); }
.estado-moviendo   { background: rgba(255,184,0,0.15);  color: var(--amber); border: 1px solid rgba(255,184,0,0.3); }
.estado-emergencia { background: rgba(255,59,92,0.15);  color: var(--red);   border: 1px solid rgba(255,59,92,0.3); }

.info-card {
    background: var(--panel);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border: 1px solid #1E2D45;
}
.info-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--fog);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.log-ok    { color: var(--green);  font-size: 0.82rem; padding: 0.2rem 0; }
.log-error { color: var(--red);    font-size: 0.82rem; padding: 0.2rem 0; }
.log-info  { color: var(--silver); font-size: 0.82rem; padding: 0.2rem 0; }
.log-time  { color: var(--fog);    font-size: 0.72rem; margin-right: 0.4rem; }

.result-bubble {
    background: var(--panel);
    border: 1px solid var(--cyan);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-size: 0.92rem;
    color: var(--white);
    margin-top: 0.8rem;
}
.result-bubble span.label {
    font-size: 0.7rem;
    color: var(--cyan);
    font-weight: 700;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 0.2rem;
}
</style>
"""


# ══════════════════════════════════════════════════════════════
# HELPERS DE UI
# ══════════════════════════════════════════════════════════════
def estado_css():
    e = st.session_state.estado
    if e == "disponible":
        return "estado-disponible", "● DISPONIBLE"
    elif e == "moviendo":
        return "estado-moviendo", "▲ EN MOVIMIENTO"
    else:
        return "estado-emergencia", "🚨 EMERGENCIA"


def render_hud():
    css_class, label = estado_css()
    destino_txt = (
        f"→ Piso {st.session_state.piso_destino}"
        if st.session_state.piso_destino
        else "En espera"
    )
    st.markdown(f"""
    <div class="hud-bar">
        <div>
            <div class="hud-label">Piso actual</div>
            <div class="hud-piso">{st.session_state.piso_actual}</div>
        </div>
        <div style="text-align:center">
            <div class="hud-label">Estado</div>
            <div class="estado-pill {css_class}">{label}</div>
        </div>
        <div style="text-align:right">
            <div class="hud-label">Destino</div>
            <div style="font-size:1.4rem;font-weight:700;color:#CBD5E1">{destino_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_log():
    if not st.session_state.log:
        return
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<div class="info-card-title">📋 Historial de eventos</div>', unsafe_allow_html=True)
    for entry in reversed(st.session_state.log[-8:]):
        st.markdown(
            f'<div class="log-{entry["tipo"]}">'
            f'<span class="log-time">{entry["t"]}</span>{entry["msg"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VozElevate",
    page_icon="🛗",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛗 VozElevate")
    st.markdown("---")
    st.markdown("### 👆 Seleccionar piso")

    cols = st.columns(2)
    for i, piso in enumerate(PISOS_VALIDOS):
        col = cols[i % 2]
        with col:
            disabled = (
                st.session_state.estado in ("moviendo", "emergencia")
                or piso == st.session_state.piso_actual
            )
            if st.button(f"Piso {piso}", key=f"btn_piso_{piso}",
                         disabled=disabled, use_container_width=True):
                mover_ascensor(piso)
                st.rerun()

    st.markdown("---")
    st.markdown("### 🚨 Emergencia")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚨 Activar", use_container_width=True):
            activar_emergencia()
            st.rerun()
    with c2:
        if st.button("✅ Cancelar", use_container_width=True):
            cancelar_emergencia()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748B'>Streamlit + Wokwi + MQTT<br>Proyecto Final · 2025</small>",
        unsafe_allow_html=True,
    )


# ── Título y HUD ─────────────────────────────────────────────
st.markdown("## 🛗 VozElevate — Ascensor Inteligente Multimodal")
render_hud()

tab_voz, tab_dibujo, tab_log = st.tabs(["🎙️ Voz", "✏️ Dibujo", "📋 Historial"])


# ════════════════════════════════════════════════════════════
# TAB 1 — CONTROL POR VOZ
# ════════════════════════════════════════════════════════════
with tab_voz:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card-title">Instrucciones</div>'
        '<p style="color:#CBD5E1;font-size:0.88rem;margin:0">'
        'Presiona <strong>Grabar</strong>, habla y presiona <strong>Detener</strong>.<br>'
        'Ejemplos: '
        '<code style="background:#0A0F1E;padding:2px 6px;border-radius:6px">"ir al piso 4"</code> · '
        '<code style="background:#0A0F1E;padding:2px 6px;border-radius:6px">"sube al tres"</code> · '
        '<code style="background:#0A0F1E;padding:2px 6px;border-radius:6px">"dos"</code>'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Grabador de micrófono
    audio = mic_recorder(
        start_prompt="🎙️  Iniciar grabación",
        stop_prompt="⏹️  Detener y reconocer",
        just_once=True,
        use_container_width=True,
        key="mic_voz",
    )

    if audio and audio.get("bytes"):
        with st.spinner("🧠 Reconociendo audio..."):
            texto_reconocido = reconocer_audio(audio["bytes"])

        if texto_reconocido:
            st.markdown(f"""
            <div class="result-bubble">
                <span class="label">VOZ RECONOCIDA</span>
                "{texto_reconocido}"
            </div>
            """, unsafe_allow_html=True)

            piso = extraer_piso_de_texto(texto_reconocido)

            if piso:
                st.success(f"🎯 Piso identificado: **{piso}**")
                agregar_log(f'Voz: "{texto_reconocido}" → Piso {piso}', "ok")
                mover_ascensor(piso)
                st.rerun()
            else:
                st.warning("⚠️ No se identificó un piso válido (1–6). Intenta de nuevo.")
                agregar_log(f'Voz sin piso válido: "{texto_reconocido}"', "error")
        else:
            st.error("❌ No se pudo entender el audio. Habla más cerca del micrófono.")
            agregar_log("Audio no reconocido", "error")

    # Comandos de referencia
    st.markdown("---")
    st.markdown("**Comandos de ejemplo:**")
    ejemplos = [
        ("⬆️", "subir al piso 5"),
        ("⬇️", "bajar al dos"),
        ("🎯", "ir al cuatro"),
        ("🏠", "primer piso"),
        ("6️⃣", "sexto piso"),
        ("🚨", "emergencia"),
    ]
    cols = st.columns(3)
    for i, (icon, cmd) in enumerate(ejemplos):
        cols[i % 3].markdown(
            f'<div class="info-card" style="text-align:center;padding:0.6rem">'
            f'<div style="font-size:1.3rem">{icon}</div>'
            f'<div style="font-size:0.78rem;color:#94A3B8;margin-top:0.2rem"><em>"{cmd}"</em></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# TAB 2 — CONTROL POR DIBUJO
# ════════════════════════════════════════════════════════════
with tab_dibujo:
    col_canvas, col_resultado = st.columns([1, 1], gap="large")

    with col_canvas:
        st.markdown('<div class="info-card-title">✏️ Dibuja el número del piso (1–6)</div>',
                    unsafe_allow_html=True)

        grosor = st.slider("Grosor del trazo", 8, 30, 18, key="grosor_canvas")

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=grosor,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=220,
            width=220,
            drawing_mode="freedraw",
            key="canvas_digito",
            display_toolbar=True,
        )

        c1, c2 = st.columns(2)
        boton_predecir = c1.button("🔍 Reconocer", use_container_width=True, type="primary")
        boton_limpiar  = c2.button("🗑️ Limpiar",   use_container_width=True)

    with col_resultado:
        st.markdown('<div class="info-card-title">🤖 Resultado del reconocimiento</div>',
                    unsafe_allow_html=True)

        if boton_predecir:
            if canvas_result.image_data is not None:
                datos = canvas_result.image_data
                pixels_con_contenido = np.sum(datos[:, :, 3] > 10)

                if pixels_con_contenido < 50:
                    st.warning("⚠️ El canvas está vacío. Dibuja primero un número.")
                else:
                    with st.spinner("Analizando dibujo..."):
                        digito = predecir_digito(datos)

                    if digito is None:
                        st.info(
                            "ℹ️ Modelo CNN no disponible.\n\n"
                            "Sube `model/handwritten.h5` al repositorio, "
                            "o usa los botones del panel lateral."
                        )
                        agregar_log("CNN no disponible", "error")
                    elif digito < 1 or digito > 6:
                        st.error(
                            f"❌ Se reconoció **{digito}**, pero solo se aceptan pisos 1–6.\n\n"
                            "Intenta dibujar más claro."
                        )
                        agregar_log(f"Dibujo: dígito {digito} fuera de rango", "error")
                    else:
                        st.success(f"✅ Número reconocido: **{digito}**")
                        st.markdown(f"""
                        <div style="font-size:4rem;font-weight:700;color:#00E5FF;
                                    text-align:center;padding:1rem;background:#111827;
                                    border-radius:16px;margin:0.5rem 0">{digito}</div>
                        """, unsafe_allow_html=True)
                        agregar_log(f"Dibujo reconocido: Piso {digito}", "ok")

                        if st.button(f"🛗 Ir al Piso {digito}", type="primary",
                                     use_container_width=True):
                            mover_ascensor(digito)
                            st.rerun()
            else:
                st.warning("⚠️ Dibuja un número antes de reconocer.")
        else:
            st.markdown("""
            <div style="background:#111827;border:2px dashed #1E2D45;border-radius:16px;
                        padding:2.5rem;text-align:center;color:#64748B">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">✏️</div>
                <div style="font-weight:600">Dibuja un número<br>y presiona <em>Reconocer</em></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">ℹ️ Cómo funciona</div>
            <ul style="color:#94A3B8;font-size:0.82rem;margin:0;padding-left:1.2rem">
                <li>Dibuja el número del piso en el canvas negro</li>
                <li>El modelo CNN (entrenado con MNIST) reconoce el dígito</li>
                <li>Solo se acepta del <strong style="color:#00E5FF">1 al 6</strong></li>
                <li>Presiona <em>Reconocer</em> para analizar</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL
# ════════════════════════════════════════════════════════════
with tab_log:
    if not st.session_state.log:
        st.info("Aún no hay eventos. Usa el ascensor para ver el historial aquí.")
    else:
        render_log()
        if st.button("🗑️ Limpiar historial"):
            st.session_state.log = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Resumen de sesión")
    total_ok    = sum(1 for e in st.session_state.log if e["tipo"] == "ok")
    total_error = sum(1 for e in st.session_state.log if e["tipo"] == "error")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Piso actual",      st.session_state.piso_actual)
    c2.metric("Viajes exitosos",  total_ok)
    c3.metric("Alertas",          total_error)
    c4.metric("Total eventos",    len(st.session_state.log))
