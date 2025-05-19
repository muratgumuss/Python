import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import json
import re
import streamlit as st


# Load the IMDB dataset word index
#word_index = imdb.get_word_index()
with open('word_index.json', 'r') as f:
    word_index = json.load(f)
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model with relu activation
model = load_model('simple_rnn_imdb_relu.h5')

# Step 2: Helper function to decode reviews
def decode_review(encoded_review):
    """
    Decode the review text from integers to words.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess the input data
def preprocess_input_data(reviews, maxlen=500):
    """
    Preprocess the input data by padding sequences.
    """
    # İşlemi her bir yorum için ayrı ayrı yap
    encoded_reviews = []
    for review in reviews:
        # Noktalama işaretlerini kaldır ve küçük harfe dönüştür
        review = re.sub(r'[^\w\s]', '', review.lower())
        words = review.split()
        encoded_review = [word_index.get(word, 0) + 3 for word in words]
        encoded_reviews.append(encoded_review)
    
    # Pad sequences
    padded_reviews = sequence.pad_sequences(encoded_reviews, maxlen=maxlen)
    return padded_reviews


# Function to predict the sentiment of a review
# Prediction function
def predict_review_sentiment(review):
    """
    Predict the sentiment of a review.
    """
    # Preprocess the review
    preprocess_input = preprocess_input_data([review])
    
    # Make prediction
    prediction = model.predict(preprocess_input)
    
    # Determine sentiment
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment, prediction[0][0]


# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")
user_input = st.text_area("Movie Review")

if st.button("Predict"):
    if user_input:

        # Predict sentiment
        sentiment, score = predict_review_sentiment(user_input)
        
        # Display result
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.write("Please enter a review to get a prediction.")
