#1. Gerekli Kütüphanelerin Yüklenmesi
"""
validators: Girilen URL'nin geçerli olup olmadığını kontrol eder.

streamlit: Web uygulaması oluşturmak için kullanılan framework.

langchain.prompts.PromptTemplate: LLM için özel prompt şablonu tanımlamak için.

ChatGroq: Groq’un LLM API’sine erişim sağlayan sınıf.

load_summarize_chain: Belge özetleme zinciri kurmak için kullanılır.

UnstructuredURLLoader: URL'den veri çeken LangChain bileşeni (kullanılmamış).

YouTubeTranscriptApi: YouTube videolarından otomatik altyazı almak için.

Document: LangChain içeriğini taşımak için kullanılan temel yapı.

requests, BeautifulSoup: Web sayfalarının içeriğini çekmek ve analiz etmek için.
"""

from sqlalchemy import over
import streamlit as st
st.set_page_config(
    page_title="Advanced Content Summarizer",
    page_icon="📝",
    layout="wide"
)

import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import time
import random
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# 2. Arayüz Yapılandırması
"""
Streamlit sayfasının başlığını ve görünen başlık metnini tanımlar.
"""

#3. API Anahtar Girişi (Sidebar'da)
"""
Kullanıcının Groq API anahtarını girmesi için bir parola alanı oluşturur.

Uzun içeriklerin parçalanarak özetlenebileceği bilgisi verilir.
"""
# Sidebar for API key
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", type="password")
    st.info("For longer content, summaries may be chunked")

# 4. URL Girdisi
"""
Kullanıcının özetlemek istediği URL’yi girmesi için bir metin kutusu.
"""
url_input = st.text_input("URL", label_visibility="collapsed")

#5. Prompt Şablonu Tanımı
"""
LLM’e verilecek girdinin nasıl yapılandırılacağını tanımlar.

{text} kısmı dinamik olarak içerikle doldurulur.
"""
# Prompt template
prompt_template = """
Please provide a concise summary of the following content in about 300 words.
Focus on key points and main ideas:

Content:{text}

SUMMARY:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 6. LLM Nesnesi Oluşturma
"""
Groq üzerinden LLaMA 3 modelini çalıştıracak LLM nesnesi oluşturur.

temperature=0.3 → daha tutarlı, az yaratıcı çıktı sağlar.
"""

# Optimized LLM initialization
def get_llm():

    repo_id="google/gemma-2-9b"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=150,temperature=0.7)
    return llm

#7. Web İçeriği Çekme Fonksiyonu
"""
requests ile URL’ye istek gönderir.

BeautifulSoup ile HTML parse edilir.

article, main, div.content gibi ana içerik bloklarını arar.

Bulamazsa body içeriğinin ilk 5000 karakterini döndürür.

Temizleme işlemi ile script, style gibi gereksiz elemanlar çıkarılır.
"""
# Improved web content extractor
def get_web_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content
        for tag in ['article', 'main', 'div.content', 'div.post-content']:
            content = soup.find(tag)
            if content:
                text = content.get_text(separator='\n', strip=True)
                if len(text) > 200:
                    return Document(page_content=text)
        
        # Fallback to body
        text = soup.get_text(separator='\n', strip=True)
        return Document(page_content=text[:5000])  # Limit content size
    
    except Exception as e:
        raise RuntimeError(f"Could not fetch web content: {str(e)}")

#8. YouTube Transcript Çekme Fonksiyonu
"""
YouTube linkinden video_id çıkarılır.

YouTubeTranscriptApi.get_transcript çağrısı yapılır.

Altyazılar birleştirilir.

Hata olursa başlık ve açıklama gibi fallback metadata çekilir.
"""
# YouTube transcript fetcher
def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else video_url.split("youtu.be/")[-1].split("?")[0]
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'tr'])
            text = " ".join([entry['text'] for entry in transcript])
            return Document(page_content=text[:5000])  # Limit transcript size
        except:
            # Fallback to metadata
            response = requests.get(video_url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('meta', property='og:title')
            description = soup.find('meta', property='og:description')
            content = f"Title: {title['content'] if title else 'No title'}\n\nDescription: {description['content'] if description else 'No description'}"
            return Document(page_content=content)
            
    except Exception as e:
        raise RuntimeError(f"YouTube error: {str(e)}")

#9. Uzun İçerikler İçin Parçalı Özetleme
"""
2000 karakterlik parçalara ayırır.

Her parçayı load_summarize_chain ile özetler.

Aralara time.sleep(1) koyarak rate-limit riskini azaltır.

Özetleri birleştirip tek bir çıktı olarak döndürür.
"""
# Chunked summarization
def summarize_large_content(docs):
    llm = get_llm()
    
    # Split large content
    content = docs[0].page_content
    chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
    
    # Process each chunk
    summaries = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        progress_bar.progress((i + 1) / len(chunks))
        chunk_doc = Document(page_content=chunk)
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain.run([chunk_doc])
        summaries.append(summary)
        time.sleep(1)  # Rate limit control
    
    return "\n\n".join(summaries)
#10. URL İşleme (Tetikleme Fonksiyonu)
"""
YouTube ise get_youtube_transcript, değilse get_web_content çağırılır.

İçerik uzun ise parçalayıp özetler, değilse direkt özet zinciri çalıştırır.

Hata oluşursa yakalanır ve kullanıcıya bildirilir.
"""

# Main processing function
def process_url(url):
    try:
        with st.spinner("Extracting content..."):
            if "youtube.com" in url or "youtu.be" in url:
                docs = [get_youtube_transcript(url)]
            else:
                docs = [get_web_content(url)]
                
            if not docs or not docs[0].page_content.strip():
                raise ValueError("No content found")
            
            with st.spinner("Summarizing..."):
                if len(docs[0].page_content) > 2000:
                    return summarize_large_content(docs)
                else:
                    chain = load_summarize_chain(get_llm(), chain_type="stuff", prompt=prompt)
                    return chain.run(docs)
                    
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        raise
#11. UI – “Summarize” Butonunun Davranışı
"""
Kullanıcı butona bastığında tetiklenir.

API anahtarı veya URL eksikse hata verilir.

Özetleme yapılır, sonuç kullanıcıya sunulur.

Ek olarak, orijinal içerik “View extracted content” alanında gösterilir.
"""
# Streamlit UI handler
if st.button("Summarize Content"):
    if not hf_api_key.strip():
        st.error("Please enter your Groq API key")
    elif not validators.url(url_input):
        st.error("Please enter a valid URL")
    else:
        try:
            result = process_url(url_input)
            st.subheader("Summary")
            st.success(result)
            
            with st.expander("View extracted content"):
                st.text_area("Content", get_web_content(url_input).page_content[:5000], height=300)
                
        except Exception as e:
            st.error(f"Failed to summarize: {str(e)}")


    