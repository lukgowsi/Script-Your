import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print("Error downloading NLTK resources:", e)

# Load the dataset
themes_file_path = 'datasets/Bible_Themes.csv'
themes_df = pd.read_csv(themes_file_path)

# Extract themes and relevant information
themes = themes_df['Themes'].tolist()
books = themes_df['Book'].tolist()
events = themes_df['Event'].tolist()
chapters = themes_df['Chapters'].tolist()

# Tokenize and preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Preprocess themes
preprocessed_themes = [preprocess_text(theme) for theme in themes]

# Vectorize themes using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix_themes = vectorizer.fit_transform(preprocessed_themes)

def TFIDF_query(query, tfidf_matrix, num_recommendations=20):
    query_vector = vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_indices = cosine_similarities.argsort()[::-1]
    return most_similar_indices[:num_recommendations]

def recommend_books_and_events(query):
    # Find top 20 similar themes based on the query
    theme_indices = TFIDF_query(query, tfidf_matrix_themes)
    
    # Get books and events for the themes
    recommended_books_events = []
    for index in theme_indices:
        book = books[index]
        event = events[index]
        chapter_list = chapters[index]
        recommended_books_events.append((book, event, chapter_list))
    
    return recommended_books_events

# Get user input
query = input("Enter your query: ")

# Get recommendations based on the user's query
recommended_books_events = recommend_books_and_events(query)

# Print recommended books and events
print("\nRecommended Books and Events:")
for i, (book, event, chapter_list) in enumerate(recommended_books_events, 1):
    print(f"{i}. {book}, Event: {event}, Chapters: {chapter_list}")
