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

# Download necessary NLTK datasj
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("qa_dataset.csv", encoding='unicode_escape')
questions_list = df['Questions'].tolist()
answers_list = df['Answers'].tolist()

# Text preprocessing functions
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def preprocess_with_stopwords(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Vectorize questions
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess(q) for q in questions_list])

# Define chatbot response function
def get_response(text):
    processed_text = preprocess_with_stopwords(text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X)
    max_similarity = np.max(similarities)

    if max_similarity > 0.6:
        high_similarity_questions = [q for q, s in zip(questions_list, similarities[0]) if s > 0.6]
        target_answers = [answers_list[questions_list.index(q)] for q in high_similarity_questions]
        Z = vectorizer.fit_transform([preprocess_with_stopwords(q) for q in high_similarity_questions])
        final_similarities = cosine_similarity(vectorizer.transform([processed_text]), Z)
        closest = np.argmax(final_similarities)
        return target_answers[closest]
    else:
        return "I can't answer this question."

# Set up Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    bot_reply = get_response(user_message)
    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(debug=True)
