import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Cargar el modelo de IA
llm = OllamaLLM(model="mistral")

# Inicializar memoria
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Definir la plantilla de chat de IA
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="previous conversation: {chat_history}\nUser: {question}\nAi:"
)

# Función para ejecutar el chat con memoria
def run_chain(question):
    chat_history_text = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in st.session_state.chat_history.messages
    ])

    # Obtener respuesta de la IA
    response = llm.invoke(
        prompt.format(chat_history=chat_history_text, question=question)
    )

    # Añadir mensajes al historial
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Interfaz de Streamlit
st.title("☠️ Ai ChatBoot con memoria")
st.write("Pregunta lo que sea")

user_input = st.text_input("Tu pregunta")
if user_input:
    response = run_chain(user_input)
    st.write(f"**Tú:** {user_input}")
    st.write(f"**IA:** {response}")

# Mostrar historial del chat
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}: {msg.content}")




########## agente IA  Con memoria 

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage  # Importar tipos de mensaje
# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="mistral")

# # Inicializar la memoria
# chat_history = ChatMessageHistory()

# # Define Ai Chat prompt
# prompt = PromptTemplate(
#     input_variables=["chat_history", "question"],  # Corregido: input_variables (no input_variable)
#     template="Previous Conversation:\n{chat_history}\nUser: {question}\nAi:"
# )

# # Función para manejar el chat con memoria
# def run_chain(question):
#     # Construir el historial de chat como texto
#     chat_history_text = "\n".join([
#         f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Ai: {msg.content}" 
#         for msg in chat_history.messages
#     ])

#     # Obtener respuesta del modelo
#     response = llm.invoke(
#         prompt.format(chat_history=chat_history_text, question=question)
#     )

#     # Añadir mensajes al historial (correctamente)
#     chat_history.add_user_message(question)  # Mensaje del usuario
#     chat_history.add_ai_message(response)    # Respuesta de la IA
#     return response

# print("\n ☠️ Ai ChatBoot")
# print("typo 'exit' to stop")

# while True:
#     user_input = input("\n 🖋️ you: ")  # Corregido: /n → \n
#     if user_input.lower() == "exit":
#         print("\n bye")  # Corregido: "bey" → "bye"
#         break

#     ai_response = run_chain(user_input)
#     print(f"Ai: {ai_response}")





########### Agentew de IA Basico#############
# from langchain_ollama import OllamaLLM

# # load AI model from Ollama

# llm = OllamaLLM(model="mistral")

# print ("\n Biemvenido a tu Agente de IA/..... Reasliza tu pregunta")


# while True:
#     question = input("your Question (or type 'exit' to stop)")
#     if question. lower() == 'exit':
#         print(" Adios")
#         break

#     response = llm.invoke(question)
#     print("\n Ai response: ", response)