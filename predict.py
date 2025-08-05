import re
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

model = tf.keras.models.load_model('models/fake_news_model.h5')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_news(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    confidence = prediction if label == "Fake" else 1 - prediction

    return label, round(confidence * 100, 2)

if __name__ == "__main__":
    article = input("Paste a news article: ")
    label, confidence = predict_news(article)
    print(f"Prediction: {label} ({confidence}% confidence)")