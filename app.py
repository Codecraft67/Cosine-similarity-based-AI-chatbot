from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Download necessary NLTK data only ongce
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Global variables
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load and preprocess data
try:
    df = pd.read_csv("qa_dataset.csv", encoding='unicode_escape')
    questions_list = df['Questions'].fillna("").tolist()
    answers_list = df['Answers'].fillna("").tolist()
except Exception as e:
    questions_list = []
    answers_list = []
    print("Error loading dataset:", str(e))

# Preprocessing functions
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    return ' '.join(stemmed)

def preprocess_with_stopwords(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    return ' '.join(stemmed)

# Prepare vectorizer and fit once
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
try:
    X = vectorizer.fit_transform([preprocess(q) for q in questions_list])
except:
    X = None

# Chatbot logic
def get_response(text):
    if not questions_list or X is None:
        return "Chatbot is not ready. Please try again later."

    processed_text = preprocess_with_stopwords(text)
    user_vector = vectorizer.transform([processed_text])
    similarities = cosine_similarity(user_vector, X)
    max_sim = np.max(similarities)

    if max_sim > 0.6:
        matched_qs = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]
        matched_ans = [answers_list[questions_list.index(q)] for q in matched_qs]
        Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in matched_qs])
        final_similarities = cosine_similarity(vectorizer.transform([processed_text]), Z)
        best_match = np.argmax(final_similarities)
        return matched_ans[best_match]
    else:
        return "I'm sorry, I couldn't find an answer to that question."

# API route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Please enter a valid question."}), 400
        bot_reply = get_response(user_message)
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"response": "An error occurred: " + str(e)}), 500

# Health check
@app.route('/', methods=['GET'])
def index():
    return "Chatbot API is running."

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
