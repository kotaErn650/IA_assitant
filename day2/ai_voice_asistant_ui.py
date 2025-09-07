import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM 


# cargar el modelo IA
llm = OllamaLLM (model="mistral")

# inicializar
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()


engine = pyttsx3.init()
engine.setProperty("rate",160)

recognizer = sr.Recognizer()

def speak(text):
    engine.say(text)
    engine.runAndWait()


def listen ():
    with sr.Microphone() as source:
        st.write("🎤 Escuchando...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"👂🏽 you said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("🙋🏽‍♂️ lo siento no se entendio ")
        return ""
    except sr.RequestError:
        st.write("📡 servicio invalido")
        return ""
    

prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template="previous conversation : {chat_history}\n{question}\n Ia"
)


def run_chain(question):
     chat_history_text =  "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
     response = llm.invoke(prompt.format(chat_history= chat_history_text, question= question))

     st.session_state.chat_history.add_user_message(question)
     st.session_state.chat_history.add_ai_message(response)

     return response


st.title("🏆 Ia voz assitente (Web UI)")
st.write("🎤 clic en el boton y habla a tu asistente!")

if st.button("🎤 Start Listening"):
    user_query = listen()
    if  user_query:
        ai_response = run_chain(user_query)
        st.write(f"**tu: {user_query}")
        st.write(f"AI**: {ai_response}")
        speak(ai_response)

st.subheader("📩 chat Historial")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()} ** : {msg.content}")

