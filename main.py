import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
# Instead of importing from transformers.models.auto, import directly from the correct modules
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM, AutoModelForCausalLM  # For models like T5, BART, etc.
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification, AutoModel  # For models like BERT, etc.
#cache classes
from transformers.models.bart.configuration_bart import BartConfig  # For BART cache
from transformers.models.t5.configuration_t5 import T5Config  # For T5 cache
from transformers.models.bert.configuration_bert import BertConfig


# Load the dataset
file_path = "en.yusufali.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all verses
df['embedding'] = df['Text'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def get_most_relevant_verse(query, top_n=1):
    """Find the most relevant verse for a given query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = [util.pytorch_cos_sim(query_embedding, verse_emb).item() for verse_emb in df['embedding']]
    top_idx = torch.tensor(similarities).topk(top_n).indices.tolist()
    return df.iloc[top_idx][['Surah', 'Ayah', 'Text']]

# Interactive chatbot
print("Quran Chatbot: Ask me anything about the Quran! (Type 'exit' to quit)")
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        print("Quran Chatbot: Goodbye!")
        break
    result = get_most_relevant_verse(query)
    print(f"Quran Chatbot: Surah {result.iloc[0]['Surah']}, Ayah {result.iloc[0]['Ayah']} - {result.iloc[0]['Text']}")