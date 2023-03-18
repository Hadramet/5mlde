import os
import config
from helpers import load_pickle, save_pickle
from preprocessing import preprocess_data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix



def model_training(X: csr_matrix, y: csr_matrix) -> MultinomialNB:
    """ Train model """

    model = MultinomialNB()
    model.fit(X, y)
    return model



def model_predict(input_data: csr_matrix, model: MultinomialNB) -> np.ndarray:
    """ Predict model """

    return model.predict(input_data)



def model_evaluation(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    """ 
    Evaluate model 
    1. Accuracy
    2. Precision
    3. Recall
    4. F1

    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def model_training_and_evaluation(X_train, X_test, y_train, y_test) -> dict:
    """ Train model and evaluate """

    model = model_training(X_train, y_train)
    prediction = model_predict(X_test, model)
    evaluation = model_evaluation(y_test, prediction)
    return {'model': model, 'evaluation': evaluation}


def train_model(data_path: str, 
        save_model: bool = True, 
        save_tv: bool = True,
        local_storage: str = config.OUTPUT_FOLDER
) -> None:
    """ Train model and save model and text vectorizer """

    extract_data = preprocess_data(data_path)
    X = extract_data['X']
    y = extract_data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = model_training_and_evaluation( X_train, X_test, y_train, y_test)
    
    os.makedirs(local_storage, exist_ok=True)
    os.makedirs(config.MODEL_FOLDER, exist_ok=True)
    os.makedirs(config.TV_FOLDER, exist_ok=True)

    if save_model:
        save_pickle(config.MODEL_PATH,model)
    if save_tv:
        save_pickle(config.TV_PATH, extract_data['tv'])



def batch_inference(input_path: str, tv=None, model=None) :
    """ Batch inference """

    if tv is None:
        tv = load_pickle(config.TV_PATH)
    if model is None:
        model = load_pickle(config.MODEL_PATH)['model']

    tv_dict = {'vectorizer': tv}
    data = preprocess_data(input_path,tv_dict , False)
    return model_predict(data['X'], model)


if __name__ == '__main__':
    from utils import download_data
    download_data()
    train_model(config.DATA_PATH)
    inference = batch_inference(config.DATA_PATH)