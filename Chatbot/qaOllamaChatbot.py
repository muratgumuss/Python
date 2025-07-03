
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question,llm,temperature,mex_tokens):
    """Generate a response from the LLM based on the question."""

    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm |output_parser
    answer = chain.invoke({"question": question})
    
    return answer

# Title of the app
st.title("Q&A Chatbot with Gemma3")

# Dropdown for LLM selection
llm = st.sidebar.selectbox(
    "Select and Ollama model",
    ["gemma3"]
)

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value= 1.0, value= 0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value= 100, max_value= 2000, value= 500)

# Main interface for user input
st.title("Ask a question")
user_input = st.text_input("You:")

if user_input:
    with st.spinner("Generating response..."):
        try:
            answer = generate_response(user_input, llm, temperature, max_tokens)
            st.text_area("Answer:", value=answer, height=300)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.error("Please enter your question.")
