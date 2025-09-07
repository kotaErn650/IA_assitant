import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader  # Usamos pypdf en lugar de PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document

# Carga del modelo LLM
llm = OllamaLLM(model="mistral")

# Corrección del nombre del modelo
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inicialización de FAISS
index = faiss.IndexFlatL2(384)
vector_store = {}
summery_text = ""

# Extraer texto del PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Almacenar en FAISS
def store_in_faiss(text, filename):
    global index, vector_store
    st.write(f"📁 Storing document '{filename}' in FAISS...")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)

    vectors = embedding.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    index.add(vectors)
    vector_store[len(vector_store)] = (filename, texts)

    return "✅ Document stored successfully!"

def generate_summary(text):
    global summary_text
    st.write("generating Ai Summary...")
    summary_text = llm.invoke(f"Summarize the foloowin g document : \n\n{text[:3000]}")
    return summary_text

# Buscar y responder preguntas
def retrieve_and_answer(query):
    global index, vector_store

    query_vector = np.array(embedding.embed_query(query), dtype=np.float32).reshape(1, -1)

    D, I = index.search(query_vector, k=2)

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += "".join(vector_store[idx][1]) + "\n\n"

    if not context:
        return "⚠️ No hay data relevante."

    return llm.invoke(f"Based on the following document context, answer the question:\n\n{context}\n\nQuestion: {query}\n\nAnswer:")

##funcion de descarga del documento
def download_summary():
    if summary_text:
        st.download_button(
            label="⚫Dowload Summary",
            data=summary_text,
            file_name="Ai_Summary.txt",
            mime="text/plain"
        )

# Interfaz Streamlit
st.title("🧠 AI Document Reader")
# st.write("Sube un documento PDF y haz preguntas basadas en su contenido.")
st.write("Upload a pdf and get an Ai-generated summary ")

uploaded_file = st.file_uploader("📄 Sube un documento PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

    #generar resumen de IA
    summary = generate_summary(text)
    st.subheader("Ai-Genet=rated Summary")
    st.write(summary)

    # habilitar la descarga de Ar=chivos
    download_summary()
    

query = st.text_input("❓ Haz una pregunta basada en el documento subido")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("🤖 Respuesta de la IA:")
    st.write(answer)
