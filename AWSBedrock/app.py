
import os
import boto3
import streamlit as st


#boto3: AWS hizmetleriyle Python üzerinden iletişim kurmayı sağlayan kütüphane. Burada Bedrock servisine bağlanmak için kullanılıyor.

#streamlit: Kullanıcı arayüzünü (UI) oluşturmak için kullanılan framework. Butonlar, metin kutuları ve başlıklar gibi bileşenleri oluşturur.

# langchain_community: LangChain, farklı dil modellerini, veri kaynaklarını ve diğer bileşenleri birbirine bağlayan bir framework'tür. Bu kodda, embedding ve LLM modellerini Bedrock üzerinden kullanmak için gerekli modüller buradan import ediliyor.

# Bedrock Client Setup: boto3 ile eu-west-2 (Londra) bölgesindeki Bedrock hizmetine bağlanılır. Ardından, BedrockEmbeddings sınıfı kullanılarak, metinleri embedding'lere 

# We will using Titan Embedding model to generate embeddings
from click import prompt
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Vector Embeddings and Vector Store
from langchain.vectorstores import FAISS
# LLM Models
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseMessage, HumanMessage

# Bedrock client setup
bedrock = boto3.client('bedrock-runtime', region_name='eu-west-2')
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)

# Bu fonksiyon, uygulamanın kullanacağı veriyi hazırlar.

# PyPDFDirectoryLoader: Projenizin data klasöründeki tüm PDF dosyalarını yükler.

# RecursiveCharacterTextSplitter: Yüklenen büyük PDF dosyalarını, chunk_size (parça boyutu) ve chunk_overlap (parçalar arası çakışma) parametrelerine göre küçük 
# parçalara ayırır. Bu, RAG için çok önemlidir, çünkü bir soruya cevap ararken tüm PDF'i değil, sadece en alakalı küçük metin parçalarını kullanırız.

# Data Ingestion Veri İşleme (Data Ingestion)
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # In our testing character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embeddings and Vector Store (Vektör Deposu Oluşturma)
# Bu fonksiyon, data_ingestion fonksiyonundan gelen metin parçalarını işler.

# FAISS.from_documents: Metin parçalarını (docs), daha önce tanımladığımız Titan Embedding modeli (bedrock_embeddings) ile sayısal vektörlere dönüştürür.

# vector_store_faiss.save_local("faiss_index"): Oluşturulan bu vektör deposunu, faiss_index adıyla yerel olarak kaydeder. 
# Böylece her seferinde PDF'leri yeniden işlemek zorunda kalmayız.

def get_vector_store(docs):
    # Create FAISS vector store
    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vector_store_faiss.save_local("faiss_index")

# Bu fonksiyonlar, kullanılacak dil modellerini hazırlar.

# get_claude_llm: Claude 3 Haiku modeli için BedrockChat sınıfını kullanır. Bu sınıf, sohbet (chat) formatındaki modellerle daha uyumludur.

# get_llama3_llm: Llama 3 modeli için Bedrock sınıfını kullanır. Her iki fonksiyonda da model_id ile hangi modelin kullanılacağı belirtilir
# ve max_tokens gibi model parametreleri ayarlanır.
def get_claude_llm():
    # Using Claude 3 Haiku model - BedrockChat kullanılmalı
    llm = BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={
            "max_tokens": 512,
        }
    )
    return llm

def get_llama3_llm():
    # Using Llama 3 model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0",
                   client=bedrock,
                   model_kwargs={
                       "max_gen_len": 512,
                       "temperature": 0.7,
                       "top_p": 0.9
                   })
    return llm
# Prompt Oluşturma
# Prompt Template: Bu, dil modeline gönderilecek olan talimat metnidir. Kodun bu kısmı, modelden nasıl bir cevap beklediğimizi belirtir.

# {context}: Buraya, sorulan soruya en çok benzeyen ve vektör deposundan çekilen metin parçaları yerleştirilir.

# {question}: Buraya ise kullanıcının girdiği soru gelir.

# Human: ve Assistant: etiketleri, modelin diyalog formatında cevap vermesini sağlar.
prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but use at least summarize
with 250 words. If you don't know the answer, just say that you don't know.
Do not try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Yanıt Üretme (Response Generation)
# Bu fonksiyon, RAG sürecini ve dil modeline sorma işlemini yönetir.

# RetrievalQA.from_chain_type: Bu, LangChain'in RAG için sunduğu hazır bir zincirdir. Kendi içinde otomatik olarak şunları yapar:
    # Kullanıcının sorusunu alır (query).
    # Bu soruyu embedding'e dönüştürür.
    # Vektör deposunda (vector_store_faiss) bu embedding'e en benzeyen metin parçalarını bulur (k=3 en benzer 3 parçayı al demek).
    # Bu parçaları, PROMPT şablonundaki {context} alanına yerleştirir.
    # Oluşan tam prompt'u dil modeline (llm) gönderir.
    # Modelden gelen cevabı döner.

# try...except Bloğu: RetrievalQA zinciri bazı modellerle uyumsuzluk yaşayabilir. Bu durum için bir "fallback" (yedek) mekanizması eklenmiş.
#  try bloğu başarısız olursa, except bloğu devreye girer ve RAG sürecini manuel olarak gerçekleştirir:
   #  Önce en alakalı 3 metin parçasını (docs) kendisi bulur.
    # Prompt'u manuel olarak oluşturur.
    # Bu prompt'u, HumanMessage formatında direkt olarak LLM'e gönderir.

def get_response_llm(llm, vector_store_faiss, query):
    # Chat modeli için RetrievalQA zinciri oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )
    
    try:
        # Get response
        response = qa_chain({"query": query})
        return response['result']
    except Exception as e:
        # Chat modeli ile uyumluluk sorunu varsa basit yaklaşım kullan
        docs = vector_store_faiss.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prompt'u manuel olarak oluştur
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        # Chat modeli için mesaj formatında gönder
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm(messages)
        
        return response.content if hasattr(response, 'content') else str(response)
    
# Streamlit Uygulamasının Ana Fonksiyonu
    # main fonksiyonu, uygulamanın kullanıcı arayüzünü ve temel mantığını içerir.
    # st.header: Uygulamanın en üstündeki başlığı belirler.
    # st.text_input: Kullanıcının soru yazması için bir metin kutusu oluşturur.
    # Sidebar: Uygulamanın sol tarafında bir yan menü oluşturur.
    # "Vectors Update" butonu: Bu butona tıklandığında, data_ingestion ve get_vector_store fonksiyonları çalışır ve PDF'ler işlenerek vektör deposu oluşturulur.
    # "Claude output" ve "Llama 3 output" butonları: Bu butonlar, kullanıcı bir soru yazdıktan sonra tıklandığında, daha önce oluşturulmuş olan vektör deposunu yükler,
    #  seçilen dil modelini (Claude veya Llama 3) çağırır ve get_response_llm fonksiyonu ile cevabı üretip ekrana yazdırır.
# Bu kod, bir geliştiricinin RAG'ı kullanarak nasıl pratik bir uygulama oluşturabileceğine dair çok güzel bir örnek sunuyor. 
# Her bir parçası, büyük ve karmaşık bir sorunu (PDF'lerle sohbet etme) yönetilebilir adımlara ayırarak çözüyor.

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")
    
    # User input
    user_question = st.text_input("Ask a question from PDF files:")
    
    with st.sidebar:
        st.title("Update or Create Vector Store")
        if st.button("Vectors Update"):
            with st.spinner("Updating vectors..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vectors updated successfully!")
   
    if st.button("Claude output"):
        with st.spinner("Processing with Claude..."):
            # Vektör deposunun var olup olmadığını kontrol et
            if os.path.exists("faiss_index"):
                try:
                    # GÜVENLİK UYARISI: Bu parametre sadece güvenilir kaynaklardan gelen 
                    # dosyalar için True yapılmalıdır
                    faiss_index = FAISS.load_local(
                        "faiss_index", 
                        bedrock_embeddings, 
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    st.error(f"Vektör deposu yüklenirken hata: {e}")
                    st.info("Lütfen önce 'Vectors Update' butonuna tıklayın.")
                    return
            else:
                st.error("Vektör deposu bulunamadı! Lütfen önce 'Vectors Update' butonuna tıklayın.")
                return
                
            llm = get_claude_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Claude response generated!")

    if st.button("Llama 3 output"):
        with st.spinner("Processing with Llama 3..."):
            # Vektör deposunun var olup olmadığını kontrol et
            if os.path.exists("faiss_index"):
                try:
                    # GÜVENLİK UYARISI: Bu parametre sadece güvenilir kaynaklardan gelen 
                    # dosyalar için True yapılmalıdır
                    faiss_index = FAISS.load_local(
                        "faiss_index", 
                        bedrock_embeddings, 
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    st.error(f"Vektör deposu yüklenirken hata: {e}")
                    st.info("Lütfen önce 'Vectors Update' butonuna tıklayın.")
                    return
            else:
                st.error("Vektör deposu bulunamadı! Lütfen önce 'Vectors Update' butonuna tıklayın.")
                return
                
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Llama 3 response generated!")

if __name__ == "__main__":
    main()


   # Bu kod, RAG (Retrieval Augmented Generation) adı verilen bir tekniği kullanarak PDF dosyalarınızla sohbet etmenizi sağlayan
   # bir web uygulaması oluşturmak için yazılmıştır. Uygulama, sorularınıza PDF'lerin içeriğine dayanarak cevap verir.
   # Bu süreci adım adım, temel kavramlardan başlayarak inceleyelim.

# 1. Temel Kavramlar

# Yazılım geçmişiniz olduğu için bu kavramları bilmeniz, kodun mantığını anlamanızı kolaylaştıracaktır:

   # Embeddings (Gömme Vektörleri): Bir metni, o metnin anlamını temsil eden bir sayı dizisine (vektöre) dönüştürme işlemidir.
   # "kedi" kelimesi ile "kedigiller" kelimesi, embedding vektörleri olarak birbirine daha yakın olurken, "kedi" ile "masa" kelimesi daha uzak konumlanır.
   # Bu, bilgisayarların metinler arasındaki anlamsal ilişkiyi anlamasını sağlar.

   # Vector Store (Vektör Deposu): Bu embedding vektörlerinin saklandığı veritabanıdır. Vektör deposu sayesinde, bir soru için en alakalı metin parçalarını
   # hızlıca bulabiliriz. Kodda FAISS adlı bir kütüphane bu amaçla kullanılıyor.

   # Retrieval Augmented Generation (RAG): Bu, bir dil modelinin (LLM) yeteneklerini artırmak için kullanılan bir yöntemdir.
   # Basitçe, modelden bir şey sormadan önce, bir veri tabanından (bizim örneğimizde PDF'ler) ilgili bilgileri alır ve bu bilgileri soruyla birlikte modele sunarız.
   # Böylece model, kendi genel bilgisi yerine, sağladığımız somut verilere dayanarak cevap verir.
