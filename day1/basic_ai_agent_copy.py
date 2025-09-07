import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="mistral")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template= "previous Conversacion : {chat_history}\n\nUser: {question}\n\n AI : "
)

def run_chain(question ):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response


#interfas grafica de Streamlit 

st.title(" 😈 Ai Caht boot con memoria")
st.write("Realiza tu Pregunta")


# paar la entrada del usuario
user_input = st.text_input("tu Pregunta : ")
if user_input:
    response = run_chain(user_input)
    st.write(f"tu pregunta fue..{user_input}")
    st.write(f"Ai Resouesta :{response}")


st.subheader("Historial del chat")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}**: {msg.content}")




                                                                                                                                                                                           
