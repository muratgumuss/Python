#1. Gerekli Kütüphaneleri Yükleme

import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

"""dotenv: Ortam değişkenlerini .env dosyasından yükler.
os: Ortam değişkenlerine erişir.
langchain_community.llms (Ollama): Ollama LLM’lerini (yerel büyük dil modeli) kullanmanı sağlar.
streamlit: Web arayüzü oluşturmak için kullanılır.
langchain_core.prompts: Prompt (talimat) şablonları oluşturur.
langchain_core.output_parsers: Model çıktısını işler.
"""
#2. Ortam Değişkenlerini Yükleme

# Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
#3. Prompt (Talimat) Şablonu Oluşturma
#Kullanıcıdan gelen soruya göre doldurulacak bir mesaj şablonu oluşturur.

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.Please answer the user's questions in a concise and informative manner."),
    ("user", "Question:{question}"),
])

#4. Streamlit Arayüzü Oluşturma
#Web arayüzünde başlık ve kullanıcıdan soru girişi için bir kutu ekler.

#Streamlit 
st.title("Ollama GenAI App")
input_text = st.text_input("Enter your question:")
#5. Kullanıcı Sorgusu Geldiğinde Modeli Çalıştırma
"""Ollama: Belirtilen modeli (gemma:2b) başlatır.
Prompt: Kullanıcıdan gelen soruyu şablona yerleştirir.
llm.invoke: Modelden cevap alır.
StrOutputParser: Modelin cevabını sade metne çevirir.
st.write: Sonucu ekranda gösterir."""
if input_text:
    # Create the LLM instance
    llm = Ollama(model="gemma:2b")

    # Prepare the prompt with user input
    prompt_with_input = prompt.format_messages(question=input_text)

    # Generate the response
    response = llm.invoke(prompt_with_input)

    # Parse the output
    output_parser = StrOutputParser()
    parsed_response = output_parser.parse(response)

    # Display the response
    st.write("Response:", parsed_response)

    """Özet
.env’den ayarları alır.
Streamlit ile web arayüzü kurar.
Kullanıcıdan soru alır, Ollama LLM ile cevap üretir ve ekranda gösterir.
Tüm adımlar zincirleme ve otomatik şekilde çalışır."""