import config
import pandas as pd
import nltk
import os

from core import preprocessing

from helpers import save_pickle
from config import TV_PATH
from prefect import task, flow
from sklearn.preprocessing import LabelEncoder



@task(name='save_label_encoder', tags=['preprocessing'])
def save_label_encoder( encoder : LabelEncoder, path: str) -> None:
    save_pickle(path, encoder)


@task(name='drop_columns', tags=['preprocessing'])
def drop_columns(df : pd.DataFrame,
                 column_to_drop: list = config.COLUMN_TO_DROP) -> pd.DataFrame:
    return preprocessing.drop_columns_core(df, column_to_drop)


@task(name='preprocess_text', tags=['preprocessing'])
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
     return preprocessing.preprocess_text(df)


@task(name='extract_x_y', tags=['preprocessing'])
def extract_x_y( df: pd.DataFrame, with_target: bool = True) -> dict:
    return preprocessing.extract_x_y(df, with_target)


@flow(name="Data processing")
def preprocess_data( path: str, with_target: bool = True,) -> dict:
    df = pd.read_csv(path)
    df = drop_columns(df)
    df = preprocess_text(df)    
    if with_target: 
        label_encoder = LabelEncoder()
        label_encoder.fit(df['label'])
        df['label'] = label_encoder.transform(df['label'])
        save_label_encoder(label_encoder,TV_PATH)
    x_y = extract_x_y(df, with_target)
    return x_y
