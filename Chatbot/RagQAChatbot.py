# RAG Q&A Conversational Chatbot With PDF including Chat History

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

#1. Kütüphanelerin ve Ortam Değişkenlerinin Yüklenmesi
"""
streamlit: Web arayüzü oluşturmak için kullanılır. Kullanıcıdan dosya ve metin girişi alır, çıktıları gösterir.
langchain ve alt modülleri: LLM (Large Language Model) tabanlı uygulamalar için zincirler, retriever’lar, prompt’lar ve mesaj geçmişi yönetimi sağlar.
Chroma: Vektör veritabanı olarak kullanılır, döküman embedding’lerini saklar ve arama/retrieval işlemlerini hızlandırır.
dotenv: .env dosyasından ortam değişkenlerini yükler (ör. API anahtarları).
os: Ortam değişkenlerini yönetmek için.
HuggingFaceEmbeddings: Metinleri vektörlere dönüştürmek için HuggingFace modellerini kullanır.
PyPDFLoader: PDF dosyalarını yükleyip metin olarak işler.
RecursiveCharacterTextSplitter: Uzun metinleri küçük parçalara böler, böylece LLM’ler ile daha verimli çalışılır.
"""
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#2. Ortam Değişkenlerinin Ayarlanması ve Embedding Modelinin Yüklenmesi
"""
HuggingFace API token’ı ortam değişkeni olarak ayarlanır.
all-MiniLM-L6-v2 modeli ile embedding işlemleri yapılır. Bu model, metinleri anlamlı vektörlere dönüştürür.
"""
st.title("RAG Q&A Conversational Chatbot with PDF")
st.write("Upload PDF files and ask questions about them.")

api_key = st.text_input("Enter your Groq API key", type="password")
#3. Streamlit Arayüzü Başlatma
"""
Başlık ve açıklama eklenir.
Kullanıcıdan Groq API anahtarı alınır (LLM erişimi için).
"""
if api_key:
    llm = ChatGroq(
        model="gemma2-9b-It",
        groq_api_key=api_key
    )

    session_id = st.text_input("Session ID", value="default_session")
#4. LLM (Dil Modeli) ve Oturum Yönetimi
    """
    Groq API anahtarı girildiyse, LLM (burada "gemma2-9b-It" modeli) başlatılır.
    Kullanıcıdan oturum (session) ID’si alınır. Böylece her kullanıcıya özel sohbet geçmişi tutulabilir.
    Streamlit’in session_state özelliği ile oturum bazlı veri saklanır.
    """
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:

        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        #5. PDF Dosyalarının Yüklenmesi ve İşlenmesi
        """
        Kullanıcıdan bir veya birden fazla PDF dosyası yüklemesi istenir.
        Her PDF dosyası geçici olarak kaydedilir ve PyPDFLoader ile metin olarak yüklenir.
        Tüm dökümanlar bir listeye eklenir."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        #6. Metin Bölme ve Vektör Veritabanı Oluşturma
        """
        Uzun metinler, LLM’in daha iyi işlemesi için küçük parçalara bölünür.
        Her parça embedding’e dönüştürülüp Chroma vektör veritabanına eklenir.
        retriever, sorulara en uygun 5 döküman parçasını bulmak için kullanılır.
        """

        #7. Soru Reformülasyonu için Prompt ve Retriever Zinciri
        """"
        Kullanıcı sorusu, önceki sohbet geçmişine referans veriyorsa, bu prompt ile bağımsız bir soruya dönüştürülür.
        Bu sayede, LLM’in soruyu daha iyi anlaması sağlanır.
        history_aware_retriever, hem döküman araması hem de soru reformülasyonu yapar."""
        contextualize_q_system_prompt = (
            "Given a chat history and a the latest user question"
            "which might refrence context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), 
                MessagesPlaceholder("chat_history"), 
                ("human", "{input}")
            ],
        )
            
        history_aware_retriever = create_history_aware_retriever(
            retriever=retriever,
            llm=llm,
            prompt=contextualize_q_prompt
        )

        # Answering the question
        """system_prompt = ("You are an assistant for question answering tasks."
                        "Use the following pieces of retrieved context to answer the question."
                        "If you don't know the answer, just say that you don't know."
                        "Use three sentences maximum to answer the question and keep it concise."
                        "\n\n"
                        "Context:\n{context}")
        """
        #8. Soru-Cevap Prompt’u ve Zinciri
        """
        Sistem prompt’u Türkçe olarak, dökümandan cevap vermesini ve bilmiyorsa “Bilmiyorum.” demesini ister.
        qa_prompt ile LLM’e hem sistem mesajı, hem sohbet geçmişi, hem de kullanıcı sorusu verilir.
        question_answer_chain ve rag_chain ile, önce dökümanlardan bilgi çekilir, sonra LLM ile cevap üretilir.
        """

        """
        Kodda iki farklı prompt/template kullanılmasının nedeni, iki farklı görevi yerine getirmek istemenizdir:

        1. contextualize_q_prompt
        Amaç: Kullanıcının sorduğu sorunun, önceki sohbet geçmişine referans verip vermediğini kontrol etmek ve gerekiyorsa bu soruyu bağımsız (tek başına anlaşılır) bir soruya dönüştürmek.
        Neden? Eğer kullanıcı “O kimdi?” gibi bir soru sorarsa, modelin önceki mesajlara bakıp “O”nun kim olduğunu anlaması gerekir. Bu prompt, önceki sohbet geçmişini ve son kullanıcı sorusunu alır, gerekirse soruyu yeniden yazar.
        Kullanıldığı yer: history_aware_retriever zincirinde, yani döküman araması yapılmadan önce.
        2. system_prompt (qa_prompt ile birlikte)
        Amaç: Modelin, dökümandan gelen bağlamı kullanarak kullanıcı sorusuna cevap vermesini sağlamak.
        Neden? Modelin, sadece dökümandaki bilgilerle ve Türkçe olarak, kısa ve öz cevap vermesi istenir. Eğer cevap dökümanda yoksa “Bilmiyorum.” demesi beklenir.
        Kullanıldığı yer: Soruya cevap üretme aşamasında (question_answer_chain zincirinde).
        Özetle
        contextualize_q_prompt: Soruyu, geçmişe referans varsa bağımsız hale getirir (retriever için).
        system_prompt (qa_prompt): Bağlamı kullanarak, kullanıcıya uygun cevap üretir (cevaplama için).
        Bu iki adım, çok daha doğru ve anlamlı cevaplar almanızı sağlar. İlki sorunun netleşmesini, ikincisi ise doğru cevabın verilmesini sağlar.
        """

        system_prompt = (
                "Sen bir soru-cevap asistanısın. "
                "Aşağıdaki bağlamı kullanarak soruya cevap ver. "
                "Eğer cevap bağlamda yoksa sadece 'Bilmiyorum.' de. "
                "Cevabın kısa ve öz olsun. "
                "\n\n"
                "Bağlam:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), 
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}")
            ],
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        """
        Çok doğru, kodda hem contextualize_q_prompt hem de system_prompt (yani qa_prompt) zincir şeklinde rag_chain içinde birleştiriliyor.
        Ama bu iki prompt zincirde farklı aşamalarda kullanılıyor:

        Akış Şöyle Çalışır:
        Kullanıcıdan Soru Alınır
        contextualize_q_prompt
        Soru, önce geçmişle birlikte history_aware_retriever'a gider.
        Burada, gerekirse soru bağımsız hale getirilir (ör: "O kimdi?" → "Atatürk kimdi?").
        Bu prompt sadece sorunun netleşmesini sağlar, cevap üretmez.
        system_prompt (qa_prompt)
        Bağımsız hale gelen soru, dökümanlardan çekilen bilgilerle birlikte LLM'e (question_answer_chain) gönderilir.
        Burada model, sadece dökümandaki bilgilerle Türkçe ve kısa bir cevap üretir.
        Yani:

        contextualize_q_prompt → Soru netleştirme (retriever aşaması)
        system_prompt (qa_prompt) → Cevap üretme (LLM aşaması)
        Her biri zincirin farklı bir halkasında görev yapar.
        Bu sayede hem geçmişe duyarlı, hem de döküman tabanlı, doğru ve kısa cevaplar alınır.

        Özet:
        İki prompt aynı zincirde ama farklı aşamalarda, farklı amaçlarla kullanılır.
        Bu, RAG mimarisinin en önemli avantajlarından biridir!
        """

        #9. Oturum Bazlı Mesaj Geçmişi Yönetimi
        """
        Her oturum için ayrı bir sohbet geçmişi tutulur.
        RunnableWithMessageHistory, hem giriş mesajlarını hem de cevapları oturum bazlı saklar.
        """
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        """
        Soru netleştirme (contextualize_q_prompt ile) ve cevap üretme (system_prompt/qa_prompt ile) aşamalarının sırası ve akışı, LangChain’in zincir (chain) fonksiyonları tarafından otomatik olarak yönetiliyor.
        Sen sadece zinciri (rag_chain) oluşturuyorsun; zincirin içinde hangi adımda hangi prompt’un kullanılacağını LangChain’in kendi kodları belirliyor.

        Yani:

        Sen zinciri kuruyorsun, adımların sırasını LangChain yönetiyor.
        Soru önce netleştiriliyor, sonra dökümanlardan bilgi çekilip cevap üretiliyor.
        Bu sıralama zincirin yapısında otomatik olarak işliyor, senin ekstra bir sıralama kodu yazmana gerek yok.
        Bu, LangChain’in zincir mimarisinin en büyük kolaylıklarından biridir!
        """
        #10. Kullanıcıdan Soru Alıp Cevaplama
        """
        Kullanıcıdan soru alınır.
        Soru, oturum geçmişiyle birlikte LLM’e gönderilir.
        Cevap ve sohbet geçmişi ekranda gösterilir.
        """
        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            
            """
            RunnableWithMessageHistory fonksiyonunda get_session_history fonksiyonunu parametre olarak veriyoruz.
            Peki, session_id parametresi bu fonksiyona nasıl geçiyor?

            Cevap:
            RunnableWithMessageHistory zinciri çalıştırılırken, zincirin .invoke() fonksiyonuna şu şekilde bir config parametresi veriyorsun:

            Buradaki config={"configurable": {"session_id": session_id}} kısmı,
            LangChain’in zincir altyapısı tarafından otomatik olarak get_session_history fonksiyonuna session_id olarak iletiliyor.

            Yani zincir çalışırken, zincirin ihtiyaç duyduğu oturum bilgisini (session_id)
            senin verdiğin config parametresinden alıp, get_session_history(session_id) fonksiyonuna otomatik olarak aktarıyor.
            Özet:
            Sen zinciri çağırırken config ile session_id’yi veriyorsun,
            LangChain bu değeri otomatik olarak get_session_history fonksiyonuna parametre olarak iletiyor.
            Bu sayede her kullanıcıya/oturuma özel sohbet geçmişi yönetilebiliyor.
            """
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History:", session_history.messages)
    else:
         st.warning("Lütfen önce PDF dosyası yükleyin.")
else:
    st.warning("Please enter your Groq API key to use the chatbot.")

"""
Özetle:
Amaç: PDF’lerden bilgi çekip, kullanıcı sorularına döküman tabanlı, çok oturumlu, geçmişe duyarlı cevaplar vermek.
Kütüphaneler: LLM, embedding, vektör arama, PDF okuma, web arayüzü ve oturum yönetimi için kullanılıyor.
Akış: PDF yükle → metinleri böl → embedding → vektör veritabanı → retriever → LLM ile cevap → sohbet geçmişiyle birlikte göster.
"""