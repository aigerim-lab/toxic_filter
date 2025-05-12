# preprocessing.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка данных
def load_data():
    train = pd.read_csv('data/train_clean.csv')
    test = pd.read_csv('data/test_clean.csv')
    test_labels = pd.read_csv('data/test_labels.csv')
    test_full = pd.merge(test, test_labels, on='id')
    test_full = test_full[test_full['toxic'] != -1]
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return train['comment_text'], train[label_columns], test_full['comment_text'], test_full[label_columns]

# TF-IDF
def get_tfidf(X_train, X_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    return vectorizer.fit_transform(X_train), vectorizer.transform(X_test)


# LSTM Tokenizer
def get_lstm_sequences(X_train, X_test, max_len=50):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    return pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len), pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
