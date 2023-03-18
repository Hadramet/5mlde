import config
import pandas as pd
import nltk
from typing import Optional
from textblob import Word
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from prefect import task, flow
from pydantic import BaseModel, BaseConfig




@task
def compute_target(
        df : pd.DataFrame,
        label: dict = config.LABEL
) -> pd.DataFrame:
    """ Compute target column from label column """

    df['target'] = df['label'].map(label)
    return df

@task
def drop_columns(
        df : pd.DataFrame,
        column_to_drop: list = config.COLUMN_TO_DROP
) -> pd.DataFrame:
    """ Drop columns """

    df = df.drop(columns=column_to_drop)
    return df

@task
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """  
    Preprocess text column 
    1. Remove punctuation
    2. Lowercase
    3. Remove stopwords
    4. Lemmatize
    """

    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.replace('[^\w\s]','')
    df['text'] = df['text'].str.lower()
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]
    freq = list(freq.index)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df

@task
def extract_x_y(
        df: pd.DataFrame,
        tv_dict: Optional[dict] = None,
        with_target: bool = True
) -> dict:
    """Extract X and y from dataframe"""

    if tv_dict is None:
        tv = TfidfVectorizer(max_features=1000,
                             lowercase=True,
                             analyzer='word',
                             stop_words='english', ngram_range=(1, 1))
        tv.fit(df['text'])
    else:
        tv = tv_dict["vectorizer"]

    X = tv.transform(df['text'])
    y = None

    if with_target:
        y = df['target']
    return {'X': X, 'y': y, 'tv': tv}

@flow
def preprocess_data(
        path: str, 
        tv_dict: Optional[dict] = None, 
        with_target: bool = True
) -> dict:
    """ Preprocess data """

    nltk.download('stopwords')
    nltk.download('wordnet')
    df = pd.read_csv(path)
    if with_target:
        df = compute_target(df)
        df = drop_columns(df)
        df = preprocess_text(df)
        return extract_x_y(df, tv_dict, with_target)
    else:
        df = preprocess_text(df)
        return extract_x_y(df, tv_dict, with_target)
