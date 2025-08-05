import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import pandas as pd

X_train_vec, X_test_vec, y_train, y_test = pd.read_pickle('data/preprocessed_data.pkl')

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_vec, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test_vec, y_test)
)

model.save('models/fake_news_model.h5')
print("Model trained and saved!")