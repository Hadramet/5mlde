from fastapi import FastAPI
import mlflow
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
import nltk
from typing import Optional,Any
from textblob import Word
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI(title="Email Spam Classifier",
              description="Email Spam Classifier API using MLFlow and FastAPI \
                to serve the model as a REST API endpoint for inference purposes only.",
              version="0.0.1")

class InputData(BaseModel):
  text: str

class ClassificationOut(BaseModel):
  label: int # 0=No-Spam , 1=Spam
  email: str 
  test : Any

pipeline = mlflow.pyfunc.load_model(model_uri="models:/email_spam/Production")

def extact_features(processed_text: str, tv_dict: Optional[dict] = {}) -> dict:
    """
    Takes a processed text
    outputs a dictionary with the extracted features.
    """
    max_features = 1000
    lowercase = True
    analyzer = 'word'
    stop_words = 'english'
    ngram_range = (1, 1)

    tv = None
    if tv_dict :
        tv = tv = tv_dict["vectorizer"]
    else:
        print("Hi")
        tv = TfidfVectorizer(
          max_features=max_features,
          lowercase=lowercase,
          analyzer=analyzer,
          stop_words=stop_words,
          ngram_range=ngram_range
        )
        tv.fit([processed_text])
    features = tv.transform([processed_text])
    
    return {"features" : features , "processed_text" : processed_text}

def process_text(payload: dict, tv_dict: Optional[dict] = {}) -> dict:
    """
    Takes a payload with a text field
    outputs a dictionary with the processed text.
    example payload:
        {'text': "Hi, I am a spam email"}
    """
    text = payload['text']
    processed_text = text.lower()
    processed_text = processed_text.replace('[^\w\s]','')
    stop = stopwords.words('english')
    processed_text = " ".join(x for x in processed_text.split() if x not in stop)
    freq = pd.Series(' '.join(processed_text).split()).value_counts()[-10:]
    freq = list(freq.index)
    processed_text = " ".join(x for x in processed_text.split() if x not in freq)
    processed_text = " ".join([Word(word).lemmatize() for word in processed_text.split()])
    features = extact_features(processed_text, tv_dict)
    return  features


def run_inference(payload: dict,
                  pipeline: Pipeline) -> dict :
    """
    Takes a pre-fitted pipeline (tfidf + naive bayes model)
    outputs the computed label.
    example payload:
        {'text': "Hi, I am a spam email"}
    """
    features = process_text(payload)
    return features

@app.get("/latest/")
def latest_model():
    return {"health_check" : "OK","run_id": pipeline._model_meta.run_id}

@app.post("/classify", response_model=ClassificationOut, status_code=201)
def classify(payload: InputData):
    label = run_inference(payload.dict(), pipeline)
    features = label["features"]
    predicted = pipeline.predict(features)
    return ClassificationOut(label=0, email=payload.text, test=predicted)
