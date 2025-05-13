from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
from preprocessing import load_data, get_tfidf

X_train, y_train, X_test, y_test = load_data()
X_train_tfidf, X_test_tfidf = get_tfidf(X_train, X_test)


nb_model = OneVsRestClassifier(MultinomialNB())
nb_model.fit(X_train_tfidf, y_train)
y_pred = nb_model.predict(X_test_tfidf)


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"✅ NB Accuracy: {acc:.4f}")
print(f"✅ NB Macro F1: {f1:.4f}")

with open('model/nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
