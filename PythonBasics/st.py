import streamlit as st
import pandas as pd
import numpy as np

st.title('Streamlit Basics')
st.write('This is a simple Streamlit app to demonstrate the basics of Streamlit.')
st.write('You can use Streamlit to create web apps for data science and machine learning.')

name = st.text_input('Enter your name:')
st.write(f'Hello {name}!')
age = st.slider('Select your age:', 0, 100, 25)
st.write(f'You are {age} years old!')

options = st.selectbox(
    'Select an option:',
    ('Option 1', 'Option 2', 'Option 3')
)
st.write(f'You selected {options}!')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.write(bytes_data)
    # To convert to a string based IO:
    string_data = bytes_data.decode('utf-8')
    st.write(string_data)
    # To read file as string:
    string_data = pd.read_csv(uploaded_file)
    st.write(string_data)
    # To read file as a dataframe:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    # To read file as a numpy array:
    numpy_data = np.genfromtxt(uploaded_file, delimiter=',')
    st.write(numpy_data)
    # To read file as a pandas dataframe:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
        
## create a dataframe
df = pd.DataFrame(
    {
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }
)
st.write('This is a dataframe:')
st.dataframe(df)
st.write(df)  # same as st.dataframe(df)
st.write('This is a table:')
st.table(df)
st.write('This is a line chart:')
st.line_chart(df)
st.write('This is a bar chart:')
st.bar_chart(df)
st.write('This is a map:')
st.map(pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [38.76, 27.4],
    columns=['lat', 'lon']
))



