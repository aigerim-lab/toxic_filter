from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import os
from preprocessing import load_data, get_tfidf

# === Настройка FastAPI и шаблонов ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Загрузка модели ===
with open("model/nb_model.pkl", "rb") as f:
    model = pickle.load(f)

# === Подготовка TF-IDF ===
X_train, _, _, _ = load_data()
X_train_tfidf, _ = get_tfidf(X_train, X_train)
tfidf_vectorizer = get_tfidf.__globals__['TfidfVectorizer'](max_features=10000)
tfidf_vectorizer.fit(X_train)

# === Метки
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# === JSON-модель для API
class Comment(BaseModel):
    text: str

# === HTML-страница
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# === POST форма (браузер)
@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request, text: str = Form(...)):
    vec = tfidf_vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    result = {label: bool(val) for label, val in zip(label_names, prediction)}
    return templates.TemplateResponse("form.html", {"request": request, "prediction": result})

# === POST API (JSON)
@app.post("/predict")
def predict_json(comment: Comment):
    vec = tfidf_vectorizer.transform([comment.text])
    prediction = model.predict(vec)[0]
    return {
        "input": comment.text,
        "prediction": {
            label: bool(val) for label, val in zip(label_names, prediction)
        }
    }
