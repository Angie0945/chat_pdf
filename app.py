import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
import numpy as np
import cv2
import tempfile
import platform
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Asistente Inteligente", page_icon="🤖", layout="centered")

# ---------------- ESTILO ----------------
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: gray;
}
.stButton>button {
    border-radius: 12px;
    background-color: #6C63FF;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🤖 Asistente Inteligente</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analiza PDFs, imágenes 🖼️ y audios 🎤</p>', unsafe_allow_html=True)

# ---------------- API KEY ----------------
api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI", type="password")

client = None
if api_key:
    client = OpenAI(api_key=api_key)

# ---------------- SUBIDA ----------------
st.markdown("### 📂 Sube tu contenido")

pdf_file = st.file_uploader("📄 PDF", type="pdf")
image_file = st.file_uploader("🖼️ Imagen", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("🎤 Audio", type=["mp3", "wav", "m4a"])

text = ""

# -------- PDF --------
if pdf_file:
    with st.spinner("📄 Leyendo PDF..."):
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""

# -------- IMAGEN --------
if image_file:
    with st.spinner("🖼️ Extrayendo texto..."):
        img = Image.open(image_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        text += pytesseract.image_to_string(img_cv)

# -------- AUDIO --------
if audio_file:
    with st.spinner("🎤 Transcribiendo audio..."):
        import speech_recognition as sr

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)

        try:
            text += recognizer.recognize_google(audio, language="es-ES")
        except:
            st.error("❌ No se pudo transcribir")

# -------- MOSTRAR TEXTO --------
if text:
    st.success("✅ Contenido listo")
    with st.expander("📜 Ver texto"):
        st.write(text[:2000] + "..." if len(text) > 2000 else text)

# -------- PREGUNTA --------
if text and client:

    st.markdown("### 💬 Haz tu pregunta")

    pregunta = st.text_input("Escribe aquí...")

    if st.button("✨ Obtener respuesta"):

        if pregunta:
            with st.spinner("🤖 Pensando..."):

                respuesta = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Responde basado SOLO en el texto proporcionado."
                        },
                        {
                            "role": "user",
                            "content": f"Texto:\n{text}\n\nPregunta:\n{pregunta}"
                        }
                    ]
                )

                st.markdown("## 🤖 Respuesta")
                st.success(respuesta.choices[0].message.content)

        else:
            st.warning("⚠️ Escribe una pregunta")

elif not api_key:
    st.info("🔑 Ingresa tu API Key")

else:
    st.info("📂 Sube contenido")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Hecho con Streamlit | Python " + platform.python_version())
