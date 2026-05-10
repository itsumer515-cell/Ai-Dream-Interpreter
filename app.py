# ── Import libraries ──────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Create the Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

# ── Step 1: Load the dataset ───────────────────────────────────────────────────
# Dataset has 902 dream symbols and their interpretations
df = pd.read_csv('dreams.csv')

# Rename columns to simple names
df.columns = ['dream', 'interpretation']

# ── Step 2: Convert BOTH symbol and interpretation into numbers using TF-IDF ──
# We combine both columns so user can describe a dream naturally
# and we match against both the symbol name AND its meaning
df['combined'] = df['dream'] + ' ' + df['interpretation']

vectorizer = TfidfVectorizer()
dream_vectors = vectorizer.fit_transform(df['combined'])

# ── Step 3: Show the homepage ──────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── Step 4: Interpret a dream ──────────────────────────────────────────────────
@app.route('/interpret', methods=['POST'])
def interpret():
    # Get the dream text the user typed
    user_dream = request.json['dream']

    # Convert user dream into numbers (same way as dataset)
    user_vector = vectorizer.transform([user_dream])

    # Compare user dream with all dreams in dataset
    similarities = cosine_similarity(user_vector, dream_vectors)

    # Find the most similar dream
    best_match_index = similarities.argmax()
    score = similarities[0][best_match_index]

    # Get the interpretation of the best match
    matched_symbol = df['dream'].iloc[best_match_index]
    interpretation = df['interpretation'].iloc[best_match_index]

    # Send result back to frontend
    return jsonify({
        'interpretation': interpretation,
        'matched_dream': matched_symbol,
        'confidence': round(float(score) * 100, 1)
    })

# ── Run the app ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
