import numpy as np
import streamlit as st
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load LSTM model
model = load_model('hamlet_lstm_model.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(input_text, model, tokenizer, max_squence_len):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([input_text])
    # Ensure the sequence length matches the model's expected input
    if len(token_list) >= max_squence_len:
        token_list = token_list[-(max_squence_len-1):]

    # Pad the sequence to the maximum length used during training
    padded_input = pad_sequences(token_list, max_squence_len - 1, padding='pre')
    # Predict the next word
    prediction = model.predict(padded_input, verbose=0)
    # Get the index of the predicted word
    predicted_word_index = np.argmax(prediction, axis=-1)
  
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# Streamlit app
st.title("Next Word Prediction with LSTM")
st.write("Enter a sentence to predict the next word:")
input_text = st.text_input("Input Text")

if st.button("Predict Next Word"):
    # Define the maximum sequence length used during training
    max_sequence_length = model.input_shape[1] + 1  # +1 for the next word prediction
    predicted_word = predict_next_word(input_text, model, tokenizer, max_sequence_length)
    
    if predicted_word:
        st.write(f"The predicted next word is: **{predicted_word}**")
    else:
        st.write("Could not predict the next word.")

