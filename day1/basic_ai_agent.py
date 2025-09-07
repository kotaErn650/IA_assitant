from langchain_ollama import OllamaLLM  
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate



# load Ai Model from Ollama


#inicializa el modelo que se esta usando
llm = OllamaLLM(model ="mistral")

#inicializar memoria
chat_history = ChatMessageHistory() # esto conserva las conversaciones con la IA
prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template="Conversacion Previa: {chat_history}\n\n User: {question}\n\nAu :}"
)

#funcuion para  memoria de IA
def run_chain(question):
    chat_history_text= "\n".join([f"{msg.type.capitaliza()}: {msg.content}" for msg in chat_history.messages])



# print("/n Bienvenido a tu agente de is")

# while True: 
#     question = input("realiza tu pregunta o escrbe 'exit para salir: ")
#     if question.lower()=='exit':
#         print("saliendo de.....")
#         break
#     response = llm.invoke(question)
#     print("/n Ai Respuesta : ", response)