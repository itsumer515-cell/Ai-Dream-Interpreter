# ── Import libraries ──────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Create the Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

# ── Step 1: Load the dataset ───────────────────────────────────────────────────
df = pd.read_csv('dreams.csv')
df.columns = ['dream', 'interpretation']

# ── Step 2: Convert dream text into numbers using TF-IDF ───────────────────────
df['combined'] = df['dream'] + ' ' + df['interpretation']
vectorizer = TfidfVectorizer()
dream_vectors = vectorizer.fit_transform(df['combined'])

# ── Step 3: Show the homepage (HTML is written directly here, no folder needed) ─
@app.route('/')
def home():
    return open('index.html').read()

# ── Step 4: Interpret a dream ──────────────────────────────────────────────────
@app.route('/interpret', methods=['POST'])
def interpret():
    user_dream = request.json['dream']
    user_vector = vectorizer.transform([user_dream])
    similarities = cosine_similarity(user_vector, dream_vectors)
    best_match_index = similarities.argmax()
    score = similarities[0][best_match_index]
    matched_symbol = df['dream'].iloc[best_match_index]
    interpretation = df['interpretation'].iloc[best_match_index]
    return jsonify({
        'interpretation': interpretation,
        'matched_dream': matched_symbol,
        'confidence': round(float(score) * 100, 1)
    })

# ── Run the app ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
