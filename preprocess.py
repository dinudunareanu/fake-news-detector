import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df = pd.read_csv('data/combined_news.csv')

df['cleaned_text'] = df['text'].apply(clean_text)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


import joblib
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
pd.to_pickle((X_train_vec, X_test_vec, y_train, y_test), 'data/preprocessed_data.pkl')

print("Preprocessing complete!")