
import pandas as pd
import nltk

from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords')
nltk.download('wordnet')
############################################
# Preprocessing
############################################


def drop_columns(df : pd.DataFrame,column_to_drop: list) -> pd.DataFrame:
    """
    Drop columns from the dataframe
    """
    df.drop(columns=column_to_drop, inplace=True)
    return df

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


def extract_x_y(
        df: pd.DataFrame,
        with_target: bool = True
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
