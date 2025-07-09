import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

#1. Kütüphanelerin İmport Edilmesi

"""
streamlit: Web arayüzü oluşturmak için kullanılır. Kullanıcıdan input almak ve çıktıları göstermek için.
os: Ortam değişkenlerine erişmek için.
langchain_groq.ChatGroq: Groq LLM API’sini kullanmak için.
langchain_openai.OpenAIEmbeddings: OpenAI’ın embedding (vektörleştirme) modelini kullanmak için.
langchain_community.embeddings.OllamaEmbeddings: Ollama ile embedding almak için (bu kodda kullanılmamış).
langchain.text_splitter.RecursiveCharacterTextSplitter: Metinleri parçalara ayırmak için.
langchain.chains.combine_documents.create_stuff_documents_chain: Belge zinciri oluşturmak için.
langchain_core.prompts.ChatPromptTemplate: LLM’e gönderilecek prompt şablonunu oluşturmak için.
langchain.chains.create_retrieval_chain: Soru-cevap için retrieval chain oluşturmak için.
langchain_community.vectorstores.FAISS: Vektör veritabanı olarak FAISS kullanmak için.
langchain_community.document_loaders.PyPDFDirectoryLoader: Bir klasördeki PDF dosyalarını yüklemek için.
dotenv.load_dotenv: .env dosyasındaki API anahtarlarını yüklemek için.
"""
from dotenv import load_dotenv
load_dotenv()

#2. Ortam Değişkenlerinin Yüklenmesi
"""
.env dosyasından API anahtarları okunur ve ortam değişkenlerine atanır.
Güvenlik için API anahtarları kodda açıkça yazılmaz.

"""
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

#3. LLM (Large Language Model) Tanımlanması
"""
ChatGroq: Groq firmasının Llama3 modelini kullanmak için.
Model ve API anahtarı belirtilir.
"""
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=groq_api_key
)
""""
Prompt template’te {context} ve {input} kullanılması, LLM tabanlı arama (RAG) sistemlerinde temel bir yaklaşımdır. Detaylı açıklayayım:

1. {context} Nedir, Neden Kullanılır?
Amaç: LLM’in (yapay zekanın) sadece ilgili belgelerden cevap vermesini sağlamak.
Nasıl çalışır: Kullanıcı bir soru sorduğunda, önce vektör veritabanında (FAISS) bu soruyla en alakalı belgeler (veya belge parçaları) aranır. Bu bulunan metinler, prompt’un {context} kısmına yerleştirilir.
Yani: {context} = Soruya en yakın bulunan belge parçalarının metni.
Örnek:
Kullanıcı: “Makine öğrenmesi nedir?”
Vektör veritabanı, PDF’lerden bu soruyla ilgili 2-3 paragraf bulur.
Bu paragraflar {context} alanına eklenir.

2. {input} Nedir, Neden Kullanılır?
Amaç: Kullanıcının sorduğu soruyu LLM’e iletmek.
Nasıl çalışır: Kullanıcıdan alınan soru, prompt’un {input} kısmına yerleştirilir.

3. Prompt Template’in Amacı
LLM’e diyor ki:
“Sadece aşağıdaki metinlere bakarak, şu soruya cevap ver.”
Böylece: Model, dışarıdan bilgi uydurmaz, sadece verdiğiniz belgelerden cevap üretir.

Akışta Ne Oluyor?
Kullanıcı soru sorar.
Vektör veritabanı, ilgili belge parçalarını bulur → {context}.
Kullanıcının sorusu → {input}.
LLM, bu iki bilgiyle prompt’u doldurur ve cevap üretir.

Kısacası:

{context}: Soruya en yakın bulunan belge metinleri
{input}: Kullanıcının sorduğu soru
Amaç: Modelin sadece elinizdeki dokümanlardan, doğru ve kaynaklı cevap vermesini sağlamak.

"""
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided  context only.
    Please provide the most accurate response based on the questions.
    <context>
    {context}
    </context>
    Question: {input}
    """
)
#5. Vektör Embedding Fonksiyonu
"""
OpenAIEmbeddings: Metinleri vektörlere dönüştürür.
PyPDFDirectoryLoader: research_papers klasöründeki PDF’leri yükler.
RecursiveCharacterTextSplitter: Belgeleri küçük parçalara böler (chunk).
FAISS: Parçaları vektör veritabanına kaydeder.
st.session_state: Streamlit’te oturum boyunca verileri saklamak için.
"""
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

"""st.session_state dinamik bir yapıdır ve içeriği kodun akışına göre belirlenir. Kodda şu satır var:

Ancak, fonksiyonun içinde st.session_state["vectors"] hiçbir yerde atanmadığı için,
 aslında bu kontrolün bir anlamı yok; çünkü "vectors" anahtarı hiçbir zaman eklenmiyor.
Yani, bu haliyle fonksiyon her çağrıldığında embedding işlemini tekrar yapar.
st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.session_state["vectors"] = True  # <-- Bu satır eklenmeli

st.session_state Nedir?
Streamlit’te oturum boyunca (sayfa yenilense bile) değişkenleri saklamak için kullanılan bir sözlüktür.
İçeriği, kodda sizin eklediğiniz anahtar-değer çiftlerinden oluşur.
Başlangıçta boştur, siz ekledikçe büyür.

"""
#6. Kullanıcıdan Soru Almak

user_prompt = st.text_input("Ask a question about the research papers")
#7. Vektör Veritabanı Oluşturma Butonu
#Butona basınca embedding işlemi başlatılır ve vektör veritabanı hazırlanır.
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")

import time

#8. Soru-Cevap ve Retrieval Chain
"""
create_stuff_documents_chain: LLM ve prompt ile bir belge zinciri oluşturur.
vectorstore.as_retriever(): Vektör veritabanından benzer belgeleri bulmak için retriever oluşturur.
create_retrieval_chain: Soru-cevap zinciri kurar (retriever + LLM).
invoke: Kullanıcının sorusunu zincire gönderir, cevap döner.
st.write: Cevap ve ilgili belgeler ekranda gösterilir.
"""
if user_prompt:
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    retriever = st.session_state.vectorstore.as_retriever()
    retriever_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()
    response = retriever_chain.invoke({
        "input": user_prompt
    })

    print(f"Response Time: {time.process_time() - start}")

    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------")

"""
PDF’ler yüklenir, parçalara ayrılır ve embedding’leri alınır.
Kullanıcıdan soru alınır.
Soruya en uygun belgeler vektör veritabanından çekilir.
Bu belgeler ve soru, LLM’e prompt ile gönderilir.
LLM’in cevabı ve ilgili belgeler ekranda gösterilir.
"""      
