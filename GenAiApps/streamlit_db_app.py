"""
Bu uygulama bir Streamlit arayüzü üzerinden, Groq LLM (örneğin LLaMA3) kullanarak doğal dilde sorular sorabileceğimiz, bu soruları SQL sorgularına çeviren ve arka plandaki bir veritabanına bağlanarak yanıtları getiren bir uygulamadır.

İki tür veritabanı destekleniyor:

    Yerel bir SQLite (student.db) dosyası

    Uzaktaki bir MySQL veritabanı
"""
from openai import api_key
import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
"""
streamlit: Web tabanlı kullanıcı arayüzü.

Path: Dosya yollarını platform bağımsız kullanmak için.

create_sql_agent: LangChain’in SQL veritabanlarıyla konuşabilen agent’ı.

SQLDatabase: SQLAlchemy motoru üzerinden DB bağlantısı.

SQLDatabaseToolkit: DB + LLM’yi araç haline getiriyor.

create_engine: SQLAlchemy motorunu oluşturur.

sqlite3: SQLite bağlantısı için.

ChatGroq: Groq API ile LLM erişimi.
"""

st.set_page_config(page_title="Langchain: Chat with Sql Db")
st.title("Langchain: Chat with Sql Db")

"""
🖼️ 2. Sayfa başlığı
Uygulama başlığını ayarlar ve başlık olarak gösterir.
"""

LocalDb = "USE_LOCALDB"
Mysql = "USE_MYSQL"

"""
3. Veritabanı Seçimi
Kullanıcıdan SQLite mi yoksa MySQL mi kullanmak istediğini seçmesini ister.

Seçime göre yapılandırma yapılır.
"""

radio_opt = ["Use sqlite 3 database - Student.db","Connect to your SQL Database"]
select_opt = st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

"""
4. Veritabanı bilgileri
Eğer 2. seçenek seçildiyse (MySQL), kullanıcıdan gerekli bilgiler istenir.
"""
if radio_opt.index(select_opt) == 1:
    db_uri = Mysql
    mysql_host = st.sidebar.text_input("Provide MySQL HOST")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password",type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri=LocalDb

api_key = st.sidebar.text_input(label="Groq Api Key", type="password")
"""
5. Groq API anahtarı
Groq API ile LLM'ye bağlanmak için gerekli anahtar.
"""
if not db_uri:
    st.info("Please enter database information and uri")

if not api_key:
    st.info("Please enter the Groq api key")

"""
6️⃣ Girdi kontrolü
Bilgiler girilmeden ilerlenirse kullanıcıya uyarı mesajı gösterilir.
"""

# llm model
"""
7️⃣ LLM Tanımı
ChatGroq nesnesi oluşturuluyor.

streaming=True demek: Model yanıtı kelime kelime verir (daha doğal bir etki yaratır).
"""
llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)

"""
8️⃣ Veritabanı yapılandırma fonksiyonu
Bu fonksiyon veritabanı bağlantısını kurar ve 2 saat boyunca cache’ler.

mode=ro: Read-only modda açar. Güvenlik açısından iyi.

SQLDatabase: Veritabanını LangChain'e uygun hale getirir.
"""
@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LocalDb:
        dbfilepath = (Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    elif db_uri==Mysql:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
"""
9️⃣ Veritabanı nesnesini oluştur

"""
if db_uri == Mysql:
    db = configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_uri)

# toolkit
"""
10. SQL Agent ve Toolkit Oluşturuluyor
Veritabanını LLM ile birlikte bir toolkit haline getiriyor.

Bu agent sayesinde LLM, doğal dil sorgusunu SQL'e dönüştürüp çalıştırabiliyor.

ZERO_SHOT_REACT_DESCRIPTION: Prompt’tan yola çıkarak ne yapması gerektiğini anlıyor.
"""
toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

"""
11. Chat geçmişi kontrolü
Kullanıcının önceki yazışmaları session_state içinde tutulur.

"Clear" butonuna basılırsa sıfırlanır.
"""
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"asistant", "content": "How can I help you?"}]

"""
12. Geçmiş mesajları göster
chat_message(...) Streamlit’in sohbet görünümü için özel olarak geliştirdiği bir bileşendir.
"""
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

"""
13. Yeni Soru Al ve Yanıtla
Kullanıcıdan gelen sorgu, doğal dilde olabilir.
"""
user_query = st.chat_input(placeholder="Ask anything from the database")

"""
14. Agent ile çalıştır
LLM gelen input’u işler → SQL üretir → SQL çalıştırılır → Yanıt doğal dile çevrilip gösterilir
"""
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        """
        Bu satır, LLM'in adım adım düşünme sürecini (LangChain "Agent" mantığında Thought, Action, Action Input, Observation gibi yapıları)
        Streamlit arayüzünde canlı olarak göstermeye yarayan bir geri çağırıcı (callback) tanımlar.

        """
        response = agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role" :"assistant", "content" :response})
        st.write(response)

"""
Neden ihtiyaç var?

LangChain'de bir Agent görev yaparken şöyle bir süreç izler:

Thought: Soruya cevap verebilmek için veritabanı sorgusu yapmalıyım.
Action: SQLDatabaseQuery
Action Input: SELECT * FROM STUDENT WHERE MARKS > 80;
Observation: 2 öğrenci bulundu.
Final Answer: Murat ve Puji 80 puanın üzerinde aldı.

İşte bu “düşünce süreci” normalde konsola (terminal) yazılır.
Ama biz bunu Streamlit arayüzünde de görmek istiyorsak, StreamlitCallbackHandler kullanmamız gerekir.

örsel takip	LLM'in adım adım nasıl düşündüğünü kullanıcıya gösterir
🔄 Callback	Agent’in her adımı bittiğinde UI’ye yazdırılır
🧩 st.container()	Yazıların nereye geleceğini belirler
✅ Kullanıcı deneyimi	Özellikle eğitim ve hata ayıklama (debugging) için çok faydalı
"""