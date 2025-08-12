import streamlit as st
import os
from dotenv import load_dotenv
import time

# LangChain + NVIDIA NIM + FAISS importları
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------------------------------
# 1) Ortam değişkenlerini yükleme
# ------------------------------------------
load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    st.error("NVIDIA_API_KEY .env dosyasında bulunamadı! Lütfen ekleyin.")
else:
    os.environ['NVIDIA_API_KEY'] = nvidia_api_key

# ------------------------------------------
# 2) Vektör DB ve embedding oluşturma
#    (cache_resource ile belleğe alınıyor)
# ------------------------------------------
@st.cache_resource
def build_vectorstore(pdf_dir="./us_census"):
    """
    PDF klasöründen dokümanları yükler, chunk'lara böler,
    embedding oluşturur ve FAISS index döner.
    """
    embeddings = NVIDIAEmbeddings()  # Embedding üretici (API çağrısı yapar)
    
    # PDF klasöründeki tüm dosyaları yükle
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()  # Document objeleri listesi (page_content + metadata)
    
    # Metni parçalama (700 karakter + 50 karakter overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs[:30])  # ilk 30 dokümanla sınırla
    
    # FAISS index oluşturma (embedding vektörleri ile)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, chunks

# ------------------------------------------
# 3) UI Başlangıcı
# ------------------------------------------
st.title("NVIDIA NIM + FAISS PDF Q&A Demo")

# LLM örneği (NVIDIA NIM üzerinden LLaMA3-70B)
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Prompt şablonu: {context} = doküman parçaları, {input} = kullanıcı sorusu
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
""")

# Kullanıcıdan soru al
user_question = st.text_input("Enter your question from the documents:")

# PDF'leri işleyip FAISS index oluşturma butonu
if st.button("Build Document Embeddings"):
    with st.spinner("Indexing documents... this may take a while..."):
        try:
            vectorstore, chunks = build_vectorstore()
            st.session_state.vectors = vectorstore
            st.success("Vector Store DB is ready ✔️")
        except Exception as e:
            st.error(f"Error during indexing: {e}")

# ------------------------------------------
# 4) Sorgulama işlemi
# ------------------------------------------
if user_question:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Build Document Embeddings' first.")
    else:
        # Document chain (stuff yöntemi ile tüm chunk'lar tek prompt'a eklenir)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # FAISS retriever (en alakalı k=4 chunk'ı getirir)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
        
        # Retriever + document chain birleşimi
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        try:
            start = time.perf_counter()  # işlem süresini ölç
            response = retrieval_chain.invoke({'input': user_question})
            elapsed = time.perf_counter() - start
            
            # Yanıtı yazdır
            st.subheader("Answer")
            st.write(response.get('answer') or "No answer returned.")
            st.caption(f"Response time: {elapsed:.2f} seconds")
            
            # Benzer doküman chunk'larını göster
            sources = response.get("context") or response.get("source_documents") or []
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.caption(str(doc.metadata))
                    st.write("---")
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
