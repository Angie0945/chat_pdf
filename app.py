import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Chat con PDF", page_icon="📄")

# ---------------- HEADER ----------------
st.title("📄 Chat inteligente con tu PDF")
st.caption("Haz preguntas y obtén respuestas basadas en el documento")

st.write("Versión de Python:", platform.python_version())

# Imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=250)
except:
    pass

# ---------------- API KEY ----------------
api_key = st.text_input("🔑 Ingresa tu API Key de OpenAI", type="password")

if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

# ---------------- PDF ----------------
pdf = st.file_uploader("📂 Sube tu PDF", type="pdf")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# ---------------- PROCESAR PDF ----------------
if pdf and api_key and st.session_state.knowledge_base is None:

    with st.spinner("Procesando PDF..."):
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20
        )

        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)

    st.success("✅ Documento listo para preguntas")

# ---------------- CHAT ----------------
if st.session_state.knowledge_base:

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input tipo chat (🔥 mejora clave)
    user_question = st.chat_input("Escribe tu pregunta sobre el PDF...")

    if user_question:
        # guardar mensaje usuario
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        # buscar info
        docs = st.session_state.knowledge_base.similarity_search(user_question)

        llm = OpenAI(temperature=0, model_name="gpt-4o-mini")

        chain = load_qa_chain(llm, chain_type="stuff")

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = chain.run(input_documents=docs, question=user_question)
                st.markdown(response)

        # guardar respuesta
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("📂 Sube un PDF y agrega tu API key para comenzar")
