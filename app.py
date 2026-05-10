# ── Import libraries ──────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Step 1: Load dataset ───────────────────────────────────────────────────────
df = pd.read_csv('dreams.csv')
df.columns = ['dream', 'interpretation']
df['dream'] = df['dream'].str.replace('\n', ' ').str.strip()

# ── Step 2: Keyword map using ONLY symbols that exist in dataset ───────────────
KEYWORD_MAP = {
    # Nature
    'water': 'Water', 'ocean': 'Ocean', 'sea': 'Sea', 'lake': 'Lake',
    'rain': 'Rain', 'rainbow': 'Rainbow', 'waves': 'Waves',
    'waterfall': 'Waterfall', 'desert': 'Desert', 'sand': 'Sand Dunes',
    'nature': 'Nature',
    'fire': 'Fire', 'flame': 'Fire', 'burning': 'Fire', 'campfire': 'Campfire',
    'snake': 'Snake', 'serpent': 'Snake',
    'moon': 'Moon', 'lunar': 'Moon',
    'forest': 'Forest', 'woods': 'Forest', 'jungle': 'Forest',
    'flying': 'Flying', 'fly': 'Flying', 'float': 'Flying',
    'chased': 'Chased', 'chase': 'Chased', 'chasing': 'Chased',
    'lost': 'Lost', 'wander': 'Lost',
    'storm': 'Storm', 'thunder': 'Storm', 'lightning': 'Storm',

    # Actions
    'fall': 'Fall', 'falling': 'Fall', 'fell': 'Fall',
    'fight': 'Fighting', 'fighting': 'Fighting',
    'walking': 'Walking', 'walk': 'Walking', 'running': 'Race',
    'waiting': 'Waiting', 'watching': 'Watching', 'laughing': 'Laughing',
    'dancing': 'Dance', 'dance': 'Dance', 'leaving': 'Leaving',
    'weeping': 'Weeping', 'crying': 'Weeping', 'screaming': 'Scream',
    'scream': 'Scream', 'washing': 'Washing',

    # People/Relations
    'father': 'Father', 'dad': 'Father', 'family': 'Family',
    'baby': 'Baby', 'child': 'Baby', 'teacher': 'Teacher',
    'man': 'Man', 'teenager': 'Teenager',

    # Animals
    'dog': 'Watchdog', 'rabbit': 'Rabbit', 'raccoon': 'Raccoon',
    'deer': 'Deer', 'rats': 'Rats', 'rat': 'Rats',
    'scorpion': 'Scorpion', 'tarantula': 'Tarantula', 'parrot': 'Parrot',
    'panda': 'Panda', 'seal': 'Seal', 'panther': 'Panther',

    # Objects
    'teeth': 'Teeth', 'tooth': 'Teeth',
    'car': 'Car', 'phone': 'Telephone', 'telephone': 'Telephone',
    'television': 'Television', 'tv': 'Television',
    'gun': 'Machine Gun', 'weapon': 'Weapons', 'weapons': 'Weapons',
    'knife': 'Machete', 'wallet': 'Wallet', 'money': 'Wealth',
    'ladder': 'Scaffolding', 'mirror': 'Magic Mirror',
    'school': 'School', 'exam': 'School',

    # States/Feelings
    'naked': 'Naked', 'nude': 'Naked',
    'scared': 'Scared', 'fear': 'Scared',
    'death': 'Death', 'dead': 'Dead', 'dying': 'Death',
    'dark': 'Darkness', 'darkness': 'Darkness',
    'late': 'Late', 'lost': 'Wandering',
    'paralyzed': 'Paralyzed', 'pain': 'Pain',
    'depression': 'Depression', 'panic': 'Panic',
    'war': 'War', 'ghost': 'Ghost', 'demon': 'Demons',

    # Places
    'castle': 'Castle', 'prison': 'Captive', 'jail': 'Captive',
    'mall': 'Mall', 'market': 'Market', 'park': 'Park',
    'temple': 'Temple', 'hospital': 'Lab Coat',
    'mansion': 'Mansion', 'backyard': 'Backyard',

    # Events
    'wedding': 'Wedding', 'party': 'Party', 'race': 'Race',
    'parade': 'Parade', 'funeral': 'Death',
    'accident': 'Accident', 'earthquake': 'Landslide',
}

# ── Step 3: Build TF-IDF as backup ────────────────────────────────────────────
df['combined'] = df['dream'] + ' ' + df['interpretation']
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
dream_vectors = vectorizer.fit_transform(df['combined'])

# ── Step 4: Homepage ───────────────────────────────────────────────────────────
@app.route('/')
def home():
    return open('index.html').read()

# ── Step 5: Interpret dream ────────────────────────────────────────────────────
@app.route('/interpret', methods=['POST'])
def interpret():
    user_dream = request.json['dream']
    words = user_dream.lower().replace(',','').replace('.','').replace('!','').replace('?','').split()

    # Priority 1: direct keyword match
    for word in words:
        if word in KEYWORD_MAP:
            target = KEYWORD_MAP[word]
            match = df[df['dream'].str.lower() == target.lower()]
            if len(match) > 0:
                row = match.iloc[0]
                return jsonify({
                    'interpretation': row['interpretation'],
                    'matched_dream': row['dream'],
                    'confidence': 95.0,
                    'source': 'dataset'
                })

    # Priority 2: TF-IDF similarity
    user_vector = vectorizer.transform([user_dream])
    similarities = cosine_similarity(user_vector, dream_vectors)
    best_index = similarities[0].argmax()
    score = float(similarities[0][best_index])

    return jsonify({
        'interpretation': df['interpretation'].iloc[best_index],
        'matched_dream': df['dream'].iloc[best_index],
        'confidence': round(score * 100, 1),
        'source': 'dataset'
    })

# ── Run app ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
