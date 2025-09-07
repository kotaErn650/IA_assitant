from langchain_ollama import OllamaLLM  

# load Ai Model from Ollama


#inicializa el modelo que se esta usando
llm = OllamaLLM(model ="mistral")


print("/n Bienvenido a tu agente de is")

while True: 
    question = input("realiza tu pregunta o escrbe 'exit para salir: ")
    if question.lower()=='exit':
        print("saliendo de.....")
        break
    response = llm.invoke(question)
    print("/n Ai Respuesta : ", response)