from utils import output
import pandas as pd
import nltk
from textblob import Word
from nltk.corpus import stopwords




def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    label = {'ham': 0, 'spam': 1}
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop('label_num', axis=1, inplace=True)
    df['label'] = df['label'].map(label)
    return df


def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    df['text'] = df['text'].astype(str)

    # Remove punctuation
    df['text'] = df['text'].str.replace('[^\w\s]','')
    
    # convert to lowercase
    df['text'] = df['text'].str.lower()

    # remove stopwords
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # remove rare words
    freq = pd.Series(' '.join(df['text']).split()).value_counts()[-10:]
    freq = list(freq.index)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # lemmatization
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df
