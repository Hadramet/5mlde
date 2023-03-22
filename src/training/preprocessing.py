import config
import pandas as pd
import nltk
import os

from helpers import save_pickle
from config import TV_PATH
from nltk.corpus import stopwords
from textblob import Word
from prefect import task, flow
from sklearn.preprocessing import LabelEncoder


@task(name='save_label_encoder', tags=['preprocessing'])
def save_label_encoder(
    encoder : LabelEncoder,
    path: str
) -> None:
    """
    Save label encoder
    Args:
        encoder (LabelEncoder): label encoder
        path (str): path to save the label encoder
    """
    save_pickle(path, encoder)


@task(name='drop_columns', tags=['preprocessing'])
def drop_columns(
        df : pd.DataFrame,
        column_to_drop: list = config.COLUMN_TO_DROP
) -> pd.DataFrame:
    """
    Drop columns from the dataframe
    """
    df.drop(columns=column_to_drop, inplace=True)
    return df



@task(name='preprocess_text', tags=['preprocessing'])
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
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


@task(name='extract_x_y', tags=['preprocessing'])
def extract_x_y(
        df: pd.DataFrame,
        with_target: bool = True,
) -> dict:
    """
    Extract X and y from dataframe
    Args:
        df (pd.DataFrame): dataframe containing text data
        with_target (bool): whether to extract target or not
    Returns:
        dict: dictionary containing X and y
    """
    if with_target:
        X = df['text']
        y = df['label']
    else:
        X = df['text']
        y = None

    return {'X': X, 'y': y }


@flow(name="Data processing")
def preprocess_data(
        path: str, 
        with_target: bool = True,
) -> dict:
    """
    Preprocess data
    Args:
        path (str): path to the data
        with_target (bool): whether to extract target or not
    Returns:
        dict: dictionary containing X and y
    """

    nltk.download('stopwords')
    nltk.download('wordnet')
    df = pd.read_csv(path)
    df = drop_columns(df)
    df = preprocess_text(df)
    
    if with_target: 
        label_encoder = LabelEncoder()
        label_encoder.fit(df['label'])
        df['label'] = label_encoder.transform(df['label'])
        save_label_encoder(label_encoder,TV_PATH)

    return extract_x_y(df,with_target)
