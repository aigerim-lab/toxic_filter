# main.py

import pandas as pd

# Загрузка данных
train = pd.read_csv('data/train_clean.csv')
test = pd.read_csv('data/test_clean.csv')
test_labels = pd.read_csv('data/test_labels.csv')

# Объединяем тестовые комментарии с их метками
test_full = pd.merge(test, test_labels, on='id')

# Убираем строки, где нет меток (-1)
test_full = test_full[test_full['toxic'] != -1]

# Выводим информацию
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("Train shape:", train.shape)
print("Test shape (after filtering):", test_full.shape)
print("Label distribution:")
print(train[label_columns].sum())
