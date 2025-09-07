import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import numpy as np
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# cargar el modelo IA
llm = OllamaLLM(model="mistral")

# inicializar el historial del Chat
chat_history = ChatMessageHistory()

# inicializar el motor de texto a voz
engine = pyttsx3.init()
engine.setProperty("rate", 160)  # ajustar la velocidad del habla

## reconocimiento de voz
recognizer = sr.Recognizer()
samplerate = 16000  # frecuencia de muestreo para sounddevice
duration = 5        # segundos de grabación

## función para hablar
def speak(text):
    engine.say(text)
    engine.runAndWait()

## función para que escuche SIN PyAudio
def listen():
    print("\n📡 Escuchando... habla ahora")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Grabación terminada")

    # Convertir a formato que entiende speech_recognition
    audio_data = sr.AudioData(audio.tobytes(), samplerate, 2)

    try:
        query = recognizer.recognize_google(audio_data, language="es-ES")
        print(f"🎤 Dijiste: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("❌ No entendí el audio")
        return ""
    except sr.RequestError:
        print("⚠️ Servicio de reconocimiento no disponible")
        return ""

# prompt de LangChain
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="previous conversation : {chat_history}\n User: {question}\n Ai:"
)

### función de proceso de respuesta
def run_chain(question):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response

# Main saludo
speak("Otra vez usted... José, ahora qué quiere, y no joda tanto")

while True:
    query = listen()
    if "salir" in query or "stop" in query:
        speak("Adiós, que tengas un buen día!")
        break
    if query:
        response = run_chain(query)
        print(f"\n🤖 IA RESPUESTA: {response}")
        speak(response)
