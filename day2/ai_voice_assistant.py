import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# cargar el modelo IA
llm = OllamaLLM (model="mistral")


# inicializar el histoprial del Chat
chat_history = ChatMessageHistory()

# inicializar el motor de texto a voz
engine = pyttsx3.init()
# ajustar la velocidad del habla
engine.setProperty("rate",160)

##reconocimiento de voz
recognizer = sr.Recognizer()

##funcion para hablar
def speak(text):
	engine.say(text)
	engine.runAndWait()
	
## funcion para que escuche
def listen():
    with sr.Microphone() as source:
        print("\n 📡 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:  # ¡Falta 'except' o 'finally'!
        query = recognizer.recognize_google(audio)
        print(f"You Said : {query}")
        return query.lower()
    except sr.UnknownValueError:
         print("Sorry no se le entende el audio")
         return""
    except sr.RequestError:
         print ("Speech sonido service no valido")
         return ""
    
prompt = PromptTemplate(
     input_variables=["chat_history", "question"],
     template= "previous conb=versation : {chat_history}\n User: {question}\n Ai:"
)



###funcion de proceso respuesta
def run_chain(question):
     chat_history_text =  "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
     # generacion de respuesta de la IS
     response = llm.invoke(prompt.format(chat_history= chat_history_text, question= question))

    #Store nueva entrada de Ia 
     chat_history.add_user_message(question)
     chat_history.add_ai_message(response)
     return response


#Main saludo
speak("Otra vez usted.. jose ahora que quiere y no joda tanto")
while True:
     query = listen()
     if "salir" in query or "stop" in query:
          speak("adios que tengas un buen dia!")
          break
     if query:
          response = run_chain(query)
          print(f"\n iA RESPUESTA : {response}")
          speak(response)  