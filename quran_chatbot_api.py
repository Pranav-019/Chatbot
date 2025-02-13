from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Load the dataset and generate embeddings once when the server starts
file_path = "en.yusufali.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all verses (cache them in the df)
df['embedding'] = df['Text'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def get_most_relevant_verse(query, top_n=1):
    """Find the most relevant verse for a given query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = [util.pytorch_cos_sim(query_embedding, verse_emb).item() for verse_emb in df['embedding']]
    top_idx = torch.tensor(similarities).topk(top_n).indices.tolist()
    return df.iloc[top_idx][['Surah', 'Ayah', 'Text']]

def generate_advice(verse_text):
    """
    Generates a short piece of advice based on the verse's content.
    """
    if "patience" in verse_text.lower():
        return "Patience is a virtue. Stay strong and trust that things will improve with time."
    elif "forgiveness" in verse_text.lower():
        return "Forgiveness brings peace to the soul. Let go of anger and embrace inner peace."
    elif "prayer" in verse_text.lower():
        return "Consistency in prayer brings clarity and peace. Stay devoted and trust in divine wisdom."
    elif "charity" in verse_text.lower():
        return "Giving to others enriches your soul. Even small acts of kindness can make a big difference."
    elif "knowledge" in verse_text.lower():
        return "Seeking knowledge leads to growth. Keep learning and expanding your understanding."
    else:
        return "Reflect on this verse deeply. It holds wisdom that may guide you in unexpected ways."

# Flask app
app = Flask(__name__)

@app.route("/get_verse", methods=["POST"])
def get_verse():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    result = get_most_relevant_verse(query)

    if result.empty:
        return jsonify({"error": "No relevant verse found"}), 404

    surah = int(result.iloc[0]['Surah'])
    ayah = int(result.iloc[0]['Ayah'])
    verse_text = result.iloc[0]['Text']
    advice = generate_advice(verse_text)

    response = {
        "description": f"Surah {surah}, Ayah {ayah}: {verse_text}",
        "advice": advice
    }
    return jsonify(response)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

