#1. Gerekli Kütüphaneleri Yükleme

"""dotenv: Ortam değişkenlerini .env dosyasından yükler.
os: Ortam değişkenlerine erişir.
fastapi: Web API oluşturmak için kullanılır.
langchain_groq: Groq LLM’lerini LangChain ile kullanmanı sağlar.
langchain_core.prompts: Prompt (talimat) şablonları oluşturur.
langchain_core.output_parsers: Model çıktısını işler.
langserve: LangChain zincirlerini FastAPI ile entegre eder.
"""

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
#2. Ortam Değişkenlerini Yükleme

from dotenv import load_dotenv
load_dotenv()

#3. Groq Modelini Oluşturma

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="gemma2-9b-It",
    api_key=groq_api_key
)

# Prompt Template
from langchain_core.prompts import ChatPromptTemplate

#4. Prompt (Talimat) Şablonu Oluşturma
#Kullanıcıdan gelen metni istenen dile çevirmek için bir şablon oluşturur.
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)
prompt_value = prompt_template.invoke({"language": "Turkish", "text": "Where is the capital of Turkey?"})
result = model.invoke(prompt_value.messages)

#5. Çıktı Parsleyicisi (Output Parser) Oluşturma
#Modelden gelen yanıtı sade metne çevirir.

parser = StrOutputParser()

#6. Zinciri Oluşturma
"""LCEL (| operatörü):
Prompt şablonunu doldurur,
Mesajı modele gönderir,
Sonucu metne çevirir.
Tüm işlemleri tek zincirde birleştirir."""
# create chain
chain = prompt_template | model | parser

#7. FastAPI Uygulamasını Tanımlama
#Web API uygulamasını başlatır.

# app definition
app = FastAPI(
    title="LangServe Groq Example",
    description="A simple FastAPI app using LangServe with Groq.",
    version="0.1.0",
)
#8. LangServe ile Route Ekleme
"""Zinciri /chain endpoint’ine bağlar.
Artık bu endpoint’e istek göndererek otomatik çeviri servisi alabilirsin."""
# add routes
add_routes(app, chain, path="/chain")

#9. Uygulamayı Çalıştırma
#Uygulamayı yerel sunucuda başlatır.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

    """.env’den Groq API anahtarını alır.
Prompt şablonu ve Groq modeliyle bir zincir kurar.
FastAPI ile bir web servisi başlatır.
/chain endpoint’ine gelen isteklerde, verilen metni istenen dile çevirip sade metin olarak döner.
Her şey otomatik ve zincirleme şekilde çalışır!"""