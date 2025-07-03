
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,mex_tokens):
    """Generate a response from the LLM based on the question."""
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm |output_parser
    answer = chain.invoke({"question": question})
    
    return answer

# Title of the app
st.title("Q&A Chatbot with OpenAI")

# Sidebar for settings
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter Your OpenAI API Key", type="password")

# Dropdown for LLM selection
llm = st.sidebar.selectbox(
    "Select and OpenAI model",
    ["gpt-4-turbo", "gpt-4","gpt-4o"]
)

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value= 1.0, value= 0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value= 100, max_value= 2000, value= 500)

# Main interface for user input
st.title("Ask a question")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        with st.spinner("Generating response..."):
            try:
                answer = generate_response(user_input, api_key, llm, temperature, max_tokens)
                st.text_area("Answer:", value=answer, height=300)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please enter your OpenAI API key.")




