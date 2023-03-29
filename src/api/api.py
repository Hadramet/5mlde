from fastapi import FastAPI
import mlflow
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
import nltk
from typing import Optional,Any
from textblob import Word
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI(title="Email Spam Classifier",
              description="Email Spam Classifier API using MLFlow and FastAPI \
                to serve the model as a REST API endpoint for inference purposes only.",
              version="0.0.1")

Instrumentator().instrument(app).expose(app)

class InputData(BaseModel):
  text: str

class ClassificationOut(BaseModel):
  label: int # 0=No-Spam , 1=Spam
  email: str 
  test : Any


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

def preprocess_text_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess text data
    Args:
        df (pd.DataFrame): dataframe containing text data
    Returns:
        pd.DataFrame: preprocessed dataframe
    """
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.replace(r'[^\w\s]', '')
    df['text'] = df['text'].str.lower()
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]
    freq = list(freq.index)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df


@app.get("/latest")
def latest_model():
  print("Health Check request received")
  pipeline = mlflow.pyfunc.load_model(model_uri="models:/email_spam_model/Production")
  print("Health Check request processed")
  return {"health_check" : "OK","run_id": pipeline._model_meta.run_id}

@app.post("/classify", response_model=ClassificationOut, status_code=201)
def classify(payload: InputData):
  print("Classification request received")
  pipeline = mlflow.pyfunc.load_model(model_uri="models:/email_spam_model/Production")
  email_content = pd.Series(payload.text)
  payload_df = pd.DataFrame({'text': email_content})
  pre_processed = preprocess_text_base(payload_df)
  predicted = pipeline.predict(pre_processed)
  print("Classification request processed")
  return ClassificationOut(label=predicted, email=payload.text, test=pre_processed)
