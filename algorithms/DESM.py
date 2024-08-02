import pandas as pd
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import numpy as np
from scipy.spatial.distance import cosine

# Download NLTK resources
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print("Error downloading NLTK resources:", e)

# Load the datasets
bible_file_path = 'datasets/bible_data_set.csv'
# themes_file_path = 'datasets/bible_themes.csv'
themes_file_path = 'datasets/bible_events.csv'

df = pd.read_csv(bible_file_path)
themes_df = pd.read_csv(themes_file_path)

# Extract text and citation
citations = df['citation'].tolist()
texts = df['text'].tolist()

# Tokenize and preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# Preprocess text
preprocessed_texts = [preprocess_text(text) for text in texts]

# Train Word2Vec model for verses
verse_model = gensim.models.Word2Vec(sentences=preprocessed_texts, vector_size=20, window=5, min_count=1, workers=4, epochs=20, seed=0)

# Preprocess and vectorize themes
themes = themes_df['Themes'].tolist()

def preprocess_and_vectorize(text):
    tokens = preprocess_text(text)
    return tokens

theme_tokens = [preprocess_and_vectorize(theme) for theme in themes]

# Train Word2Vec model for themes
theme_model = gensim.models.Word2Vec(sentences=theme_tokens, vector_size=100, window=5, min_count=1, workers=4, epochs=20, seed=0)
print(theme_model)

# Function to get the embedding of a query
def get_query_embedding(query, model):
    query_tokens = preprocess_and_vectorize(query)
    print('query tokens:', query_tokens)
    print(model.wv)
    valid_tokens = [token for token in query_tokens if token in model.wv]
    print('valid tokens:', valid_tokens)
    if not valid_tokens:
        return np.zeros(model.vector_size)  # Return a zero vector if no valid tokens
    query_embedding = np.mean([model.wv[token] for token in valid_tokens], axis=0)
    return query_embedding

# Function to find similar themes and recommend books and events
def recommend_books_events(query, num_recommendations=20):
    query_embedding = get_query_embedding(query, theme_model)
    similarity_scores = []
    
    for i, tokens in enumerate(theme_tokens):
        theme_embedding = np.mean([theme_model.wv[token] for token in tokens if token in theme_model.wv], axis=0)
        if theme_embedding is not None and query_embedding is not None:
            similarity = 1 - cosine(query_embedding, theme_embedding)
            similarity_scores.append((i, similarity))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_themes = similarity_scores[:num_recommendations]
    
    recommendations = []
    for i, _ in top_similar_themes:
        book = themes_df.iloc[i]['Book']
        event = themes_df.iloc[i]['Event']
        chapters = themes_df.iloc[i]['Chapters']
        recommendations.append((book, event, chapters))
    
    return recommendations

# Get user input
query = input("Enter your query: ")

# Get recommendations
recommended_books_events = recommend_books_events(query)

# Print recommended books and events
for i, (book, event, chapters) in enumerate(recommended_books_events, 1):
    print(f"{i}. {book}, Chapters: {chapters}, {event}")
