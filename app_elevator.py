"""
VozElevate — Ascensor Inteligente Multimodal
============================================
Modalidades de entrada:
  · Voz       → Web Speech API via Bokeh + streamlit-bokeh-events
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
import time
import json
import re
import threading

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Voz
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

# Dibujo
from streamlit_drawable_canvas import st_canvas

# MQTT
import paho.mqtt.client as paho

# Modelo CNN  (se carga una sola vez con caché)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════
BROKER   = "broker.mqttdashboard.com"
PORT     = 1883
TOPIC    = "vozelevate/cmd"
CLIENT_ID = "VozElevate_App"

PISOS_VALIDOS = list(range(1, 7))          # 1 al 6
ANGULOS_SERVO = {1: 0, 2: 30, 3: 60,       # ángulos que el Arduino espera
                 4: 90, 5: 120, 6: 150}

PALABRAS_NUMERO = {
    "uno": 1, "one": 1, "1": 1,
    "dos": 2, "two": 2, "2": 2,
    "tres": 3, "three": 3, "3": 3,
    "cuatro": 4, "four": 4, "4": 4,
    "cinco": 5, "five": 5, "5": 5,
    "seis": 6, "six": 6, "6": 6,
}

MODEL_PATH = "model/handwritten.h5"        # ruta relativa al ejecutar streamlit


# ══════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN  (persiste entre rerenders)
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "piso_actual":   1,
        "piso_destino":  None,
        "estado":        "disponible",   # disponible | moviendo | emergencia
        "log":           [],
        "mqtt_ok":       False,
        "tab_activa":    "voz",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
# MQTT
# ══════════════════════════════════════════════════════════════
def mqtt_publicar(payload: dict) -> bool:
    """Publica un mensaje JSON al broker MQTT. Retorna True si OK."""
    try:
        c = paho.Client(CLIENT_ID + "_pub")
        c.connect(BROKER, PORT, keepalive=5)
        msg = json.dumps(payload)
        result = c.publish(TOPIC, msg)
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
    # Limitar historial a 50 entradas
    if len(st.session_state.log) > 50:
        st.session_state.log = st.session_state.log[-50:]


def mover_ascensor(piso_destino: int):
    """
    Simula el movimiento piso a piso con delay visual.
    Actualiza st.session_state y envía señales MQTT.
    """
    if st.session_state.estado == "emergencia":
        st.warning("⚠️ Hay una emergencia activa. Cancélala primero.")
        return

    if piso_destino == st.session_state.piso_actual:
        agregar_log(f"Ya estás en el piso {piso_destino}.", "info")
        return

    if piso_destino not in PISOS_VALIDOS:
        agregar_log(f"Piso {piso_destino} no válido.", "error")
        return

    # ── Confirmar destino ──────────────────────────────────────
    st.session_state.piso_destino = piso_destino
    st.session_state.estado = "moviendo"
    direccion = "⬆️ Subiendo" if piso_destino > st.session_state.piso_actual else "⬇️ Bajando"

    agregar_log(f"Destino seleccionado: Piso {piso_destino}", "ok")
    agregar_log(f"{direccion} al piso {piso_destino}...", "info")

    # ── Señal MQTT: inicio de movimiento ──────────────────────
    mqtt_publicar({
        "accion": "mover",
        "piso":   piso_destino,
        "desde":  st.session_state.piso_actual,
        "angulo": ANGULOS_SERVO[piso_destino],
    })

    # ── Simular tránsito piso a piso ──────────────────────────
    paso = 1 if piso_destino > st.session_state.piso_actual else -1
    piso_tmp = st.session_state.piso_actual

    placeholder = st.empty()

    while piso_tmp != piso_destino:
        piso_tmp += paso
        st.session_state.piso_actual = piso_tmp

        with placeholder.container():
            dir_icon = "⬆️" if paso == 1 else "⬇️"
            st.info(f"{dir_icon} Pasando por piso **{piso_tmp}**...")

        time.sleep(0.9)   # 0.9 s por piso (ajustable)

    placeholder.empty()

    # ── Llegada ───────────────────────────────────────────────
    st.session_state.estado = "disponible"
    st.session_state.piso_destino = None
    agregar_log(f"✅ Llegaste al piso {piso_destino}. Puertas abiertas.", "ok")

    mqtt_publicar({
        "accion":  "llegada",
        "piso":    piso_destino,
        "angulo":  ANGULOS_SERVO[piso_destino],
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
# RECONOCIMIENTO DE VOZ  → extraer piso del texto
# ══════════════════════════════════════════════════════════════
def extraer_piso_de_texto(texto: str):
    """
    Extrae un número de piso (1-6) del texto reconocido por voz.
    Acepta: "ir al piso 4", "cuatro", "sube al 3", etc.
    Retorna int o None.
    """
    texto = texto.lower().strip()

    # Primero buscar dígito explícito
    matches = re.findall(r"\b([1-6])\b", texto)
    if matches:
        return int(matches[0])

    # Luego buscar palabras numéricas
    for palabra, num in PALABRAS_NUMERO.items():
        if palabra in texto:
            return num

    return None


# ══════════════════════════════════════════════════════════════
# RECONOCIMIENTO DE DIBUJO  → CNN MNIST → piso 1-6
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Carga el modelo CNN una sola vez y lo cachea en memoria."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def predecir_digito(imagen_rgba: np.ndarray):
    """
    Recibe el array RGBA del canvas y retorna el dígito predicho (int)
    o None si falla.
    """
    modelo = cargar_modelo()
    if modelo is None:
        return None

    try:
        img = Image.fromarray(imagen_rgba.astype("uint8"), "RGBA")
        img = img.convert("L")                  # escala de grises
        img = ImageOps.invert(img)              # fondo negro, trazo blanco (MNIST)
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
    --bg:       #0A0F1E;
    --card:     #111827;
    --panel:    #1A2235;
    --cyan:     #00E5FF;
    --teal:     #00B4CC;
    --aqua:     #00FFD1;
    --violet:   #7B2FBE;
    --lilac:    #C084FC;
    --white:    #FFFFFF;
    --silver:   #CBD5E1;
    --fog:      #64748B;
    --green:    #00FF94;
    --amber:    #FFB800;
    --red:      #FF3B5C;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Quicksand', sans-serif !important;
    color: var(--white) !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] * { color: var(--silver) !important; }

/* Tabs */
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

/* Header HUD */
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
    font-family: 'Quicksand', sans-serif;
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

/* Info card */
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

/* Log entries */
.log-ok    { color: var(--green);  font-size: 0.82rem; padding: 0.2rem 0; }
.log-error { color: var(--red);    font-size: 0.82rem; padding: 0.2rem 0; }
.log-info  { color: var(--silver); font-size: 0.82rem; padding: 0.2rem 0; }
.log-time  { color: var(--fog);    font-size: 0.72rem; margin-right: 0.4rem; }

/* Voice button override */
.bk-btn, .bk-btn-default {
    font-family: 'Quicksand', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    background: var(--cyan) !important;
    color: #050A14 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.55rem 2rem !important;
    box-shadow: 0 4px 20px rgba(0,229,255,0.35) !important;
    cursor: pointer !important;
    transition: transform 0.15s !important;
}
.bk-btn:hover { transform: scale(1.03) !important; }

/* Result bubble */
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

/* Floor button grid */
.piso-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 0.5rem;
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
# PÁGINA PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VozElevate",
    page_icon="🛗",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)


# ── Sidebar: botones de piso + controles ──────────────────────
with st.sidebar:
    st.markdown("## 🛗 VozElevate")
    st.markdown("---")
    st.markdown("### 👆 Seleccionar piso")

    cols = st.columns(2)
    for i, piso in enumerate(PISOS_VALIDOS):
        col = cols[i % 2]
        with col:
            disabled = (
                st.session_state.estado == "moviendo" or
                st.session_state.estado == "emergencia" or
                piso == st.session_state.piso_actual
            )
            if st.button(
                f"Piso {piso}",
                key=f"btn_piso_{piso}",
                disabled=disabled,
                use_container_width=True,
            ):
                mover_ascensor(piso)
                st.rerun()

    st.markdown("---")
    st.markdown("### 🚨 Emergencia")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 Activar", use_container_width=True):
            activar_emergencia()
            st.rerun()
    with col2:
        if st.button("✅ Cancelar", use_container_width=True):
            cancelar_emergencia()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748B'>Streamlit + Wokwi + MQTT<br>Proyecto Final · 2025</small>",
        unsafe_allow_html=True,
    )


# ── Contenido principal ───────────────────────────────────────
st.markdown("## 🛗 VozElevate — Ascensor Inteligente Multimodal")

render_hud()

# ── Tabs: Voz / Dibujo / Log ──────────────────────────────────
tab_voz, tab_dibujo, tab_log = st.tabs(["🎙️ Voz", "✏️ Dibujo", "📋 Historial"])


# ════════════════════════════════════════════════════════════
# TAB 1 — CONTROL POR VOZ
# ════════════════════════════════════════════════════════════
with tab_voz:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card-title">Instrucciones</div>'
        '<p style="color:#CBD5E1;font-size:0.88rem;margin:0">'
        'Presiona el botón y habla. Ejemplos:<br>'
        '<code style="background:#1A2235;padding:2px 6px;border-radius:6px">"ir al piso 4"</code> · '
        '<code style="background:#1A2235;padding:2px 6px;border-radius:6px">"sube al tres"</code> · '
        '<code style="background:#1A2235;padding:2px 6px;border-radius:6px">"baja al dos"</code>'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Botón de voz (Web Speech API)
    stt_button = Button(label="🎙  Iniciar escucha", width=240)
    stt_button.js_on_event(
        "button_click",
        CustomJS(code="""
            var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'es-ES';
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.onresult = function(e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) { value += e.results[i][0].transcript; }
                }
                if (value !== "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            };
            recognition.onerror = function(e) {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: "__error__:" + e.error}));
            };
            recognition.start();
        """),
    )

    result_voz = streamlit_bokeh_events(
        stt_button,
        events="GET_TEXT",
        key="voice_listen",
        refresh_on_update=False,
        override_height=65,
        debounce_time=0,
    )

    if result_voz and "GET_TEXT" in result_voz:
        texto_reconocido = result_voz["GET_TEXT"]

        if texto_reconocido.startswith("__error__:"):
            st.error(f"❌ Error de reconocimiento: {texto_reconocido.replace('__error__:', '')}")
        else:
            st.markdown(f"""
            <div class="result-bubble">
                <span class="label">VOZ RECONOCIDA</span>
                "{texto_reconocido}"
            </div>
            """, unsafe_allow_html=True)

            piso = extraer_piso_de_texto(texto_reconocido)

            if piso:
                st.success(f"🎯 Piso identificado: **{piso}**")
                agregar_log(f"Voz: \"{texto_reconocido}\" → Piso {piso}", "ok")
                mover_ascensor(piso)
                st.rerun()
            else:
                st.warning("⚠️ No se identificó un piso válido (1-6) en el comando.")
                agregar_log(f"Voz no reconocida: \"{texto_reconocido}\"", "error")

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
        st.markdown('<div class="info-card-title">✏️ Dibuja el número del piso (1–6)</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="info-card-title">🤖 Resultado del reconocimiento</div>', unsafe_allow_html=True)

        if boton_predecir:
            if canvas_result.image_data is not None:
                # Verificar que el canvas no esté vacío (todos transparentes/negros)
                datos = canvas_result.image_data
                pixels_con_contenido = np.sum(datos[:, :, 3] > 10)  # canal alpha

                if pixels_con_contenido < 50:
                    st.warning("⚠️ El canvas está vacío. Dibuja primero un número.")
                else:
                    with st.spinner("Analizando dibujo..."):
                        digito = predecir_digito(datos)

                    if digito is None:
                        # Modelo no disponible → modo fallback manual
                        st.info(
                            "ℹ️ Modelo CNN no disponible.\n\n"
                            "Coloca el archivo `model/handwritten.h5` en la raíz del proyecto "
                            "o usa los botones del panel lateral para seleccionar el piso."
                        )
                        agregar_log("CNN no disponible — usa botones laterales", "error")
                    elif digito < 1 or digito > 6:
                        st.error(
                            f"❌ Se reconoció el dígito **{digito}**, "
                            f"pero solo se aceptan pisos del 1 al 6.\n\n"
                            f"Intenta dibujar más claro."
                        )
                        agregar_log(f"Dibujo: dígito {digito} fuera de rango", "error")
                    else:
                        st.success(f"✅ Número reconocido: **{digito}**")
                        st.markdown(f"""
                        <div style="
                            font-size: 4rem;
                            font-weight: 700;
                            color: #00E5FF;
                            text-align: center;
                            padding: 1rem;
                            background: #111827;
                            border-radius: 16px;
                            margin: 0.5rem 0;
                        ">{digito}</div>
                        """, unsafe_allow_html=True)

                        agregar_log(f"Dibujo reconocido: Piso {digito}", "ok")

                        if st.button(f"🛗 Ir al Piso {digito}", type="primary", use_container_width=True):
                            mover_ascensor(digito)
                            st.rerun()
            else:
                st.warning("⚠️ Dibuja un número antes de reconocer.")

        else:
            # Estado vacío / inicial
            st.markdown("""
            <div style="
                background: #111827;
                border: 2px dashed #1E2D45;
                border-radius: 16px;
                padding: 2.5rem;
                text-align: center;
                color: #64748B;
            ">
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
                <li>Se acepta solo del <strong style="color:#00E5FF">1 al 6</strong></li>
                <li>Presiona <em>Reconocer</em> para analizar</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL
# ════════════════════════════════════════════════════════════
with tab_log:
    if not st.session_state.log:
        st.info("Aún no hay eventos registrados. Usa el ascensor para ver el historial.")
    else:
        render_log()

        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🗑️ Limpiar historial"):
                st.session_state.log = []
                st.rerun()

    # Resumen de sesión
    st.markdown("---")
    st.markdown("### 📊 Resumen de sesión")
    total_ok    = sum(1 for e in st.session_state.log if e["tipo"] == "ok")
    total_error = sum(1 for e in st.session_state.log if e["tipo"] == "error")
    total_info  = sum(1 for e in st.session_state.log if e["tipo"] == "info")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Piso actual",       st.session_state.piso_actual)
    c2.metric("Eventos exitosos",  total_ok)
    c3.metric("Alertas",           total_error)
    c4.metric("Total eventos",     len(st.session_state.log))
