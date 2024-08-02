import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import nltk
import ssl
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
file_path = 'bible_data_set.csv'
df = pd.read_csv(file_path)

# Extract text and citation
citations = df['citation'].tolist()
texts = df['text'].tolist()

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get the BERT embedding for a piece of text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling on token embeddings (excluding [CLS] and [SEP] tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Load or compute verse embeddings
try:
    # Try to load existing embeddings
    with open('verse_embeddings.pkl', 'rb') as f:
        verse_embeddings = pickle.load(f)
except FileNotFoundError:
    # If file not found, compute embeddings and save them
    verse_embeddings = [get_bert_embedding(verse) for verse in texts]
    with open('verse_embeddings.pkl', 'wb') as f:
        pickle.dump(verse_embeddings, f)

# Function to generate recommendations using DESM with BERT
def DESM_query(query, num_recommendations=20):
    query_embedding = get_bert_embedding(query)
    similarity_scores = []

    for i, verse_embedding in enumerate(verse_embeddings):
        similarity = cosine_similarity([query_embedding], [verse_embedding])[0][0]
        similarity_scores.append((i, similarity))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_verses = similarity_scores[:num_recommendations]
    
    return [(df.iloc[i]['citation'], df.iloc[i]['text']) for i, _ in top_similar_verses]

# Example query
query = 'anxious standing in a room of corpses'
recommended_verses = DESM_query(query)

# Print recommended verses
for i, (citation, verse) in enumerate(recommended_verses, 1):
    print(f"{i}. {citation}: {verse}")
