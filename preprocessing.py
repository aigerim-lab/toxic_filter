import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    vectorizer = TfidfVectorizer(max_features=10000)
    return vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
