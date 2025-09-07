import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_ollama import OllamaLLM

# cargar el modelo IA
llm = OllamaLLM (model="mistral")

def scrape_website(url):
    try:
        st.write(f"🌐 Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return f"⚫ Failed to fetch {url}"
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]
    
    except Exception as e:
        return f"❌ Error scraping website: {str(e)}"
    
def summarize_content(content):
    st.write("🖋️ Summarize content...")
    return llm.invoke(f"Summarize the following content : \n\n {content[1000]}")


## Streamlit site Web
st.title(" 🙋🏽IA powered Scraper")
st.write("Ingresa tu ulr 🌐💲 ")

url = st.text_input("📡 Enter tu Website")
if url:
    content = scrape_website(url)

    if"⚠️ Failed " in content or "☠️ Error "in content:
        st.write(content)
    else:
        summary = summarize_content(content)
        st.subheader("📩 website Summary")
        st.write(summary)
        

