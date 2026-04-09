import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
import numpy as np
import cv2
import tempfile
import platform

# LangChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAG Inteligente", page_icon="🤖", layout="centered")

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
st.markdown('<p class="big-title">🤖 Asistente Inteligente RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analiza PDFs, imágenes 🖼️ y audios 🎤 como un experto</p>', unsafe_allow_html=True)

st.write("🧠 **Sube contenido y haz preguntas sobre él**")

# ---------------- API KEY ----------------
api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# ---------------- SUBIDA DE ARCHIVOS ----------------
st.markdown("### 📂 Sube tu contenido")

pdf_file = st.file_uploader("📄 PDF", type="pdf")
image_file = st.file_uploader("🖼️ Imagen", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("🎤 Audio", type=["mp3", "wav", "m4a"])

text = ""

# ---------------- PROCESAR PDF ----------------
if pdf_file:
    with st.spinner("📄 Leyendo PDF..."):
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

# ---------------- PROCESAR IMAGEN ----------------
if image_file:
    with st.spinner("🖼️ Extrayendo texto de imagen..."):
        img = Image.open(image_file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        text += pytesseract.image_to_string(img_cv)

# ---------------- PROCESAR AUDIO ----------------
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
            st.error("❌ No se pudo transcribir el audio")

# ---------------- MOSTRAR TEXTO ----------------
if text:
    st.success("✅ Contenido listo para analizar")
    with st.expander("📜 Ver texto extraído"):
        st.write(text[:2000] + "..." if len(text) > 2000 else text)

# ---------------- PROCESAR RAG ----------------
if text and api_key:

    with st.spinner("🧠 Procesando conocimiento..."):
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

    st.markdown("### 💬 Haz tu pregunta")

    user_question = st.text_input("Escribe tu pregunta aquí...")

    if st.button("✨ Obtener respuesta"):
        if user_question:

            with st.spinner("🤖 Pensando..."):
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

            st.markdown("## 🤖 Respuesta")
            st.success(response)

        else:
            st.warning("⚠️ Escribe una pregunta primero")

elif not api_key:
    st.info("🔑 Ingresa tu API Key para activar el análisis")

else:
    st.info("📂 Sube al menos un archivo para comenzar")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Hecho con Streamlit + IA | Versión Python: " + platform.python_version())
