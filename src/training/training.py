import os
import config
import mlflow
import numpy as np

from utils import download_data
from helpers import task_load_pickle, task_save_pickle
from preprocessing import preprocess_data

from typing import Optional, Tuple
from scipy.sparse import csr_matrix
from prefect import task, flow
from prefect_great_expectations import run_checkpoint_validation

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@task(name='model_training', tags=['model'])
def model_training(
    X: csr_matrix, 
    y: csr_matrix, 
    **kwargs
) -> Pipeline:
    """ 
    Train the model
    Args:
        X (csr_matrix): X data
        y (csr_matrix): y data
        **kwargs: keyword arguments
    Returns:
        Pipeline: trained model
    """
    max_features = kwargs.get('max_features', 3000)
    ngram_range = kwargs.get('ngram_range', (1, 2))
    use_idf = kwargs.get('use_idf', True)
    alpha = kwargs.get('alpha', 0.01)
    analyser = kwargs.get('analyser', 'word')
    stop_words = kwargs.get('stop_words', 'english')

    

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, 
                                ngram_range=ngram_range, 
                                use_idf=use_idf, 
                                analyzer=analyser, 
                                stop_words=stop_words)),
        ('clf', MultinomialNB(alpha=alpha)),
    ])
    pipeline.fit(X, y)
    return pipeline



@task(name='model_predict', tags=['model'])
def model_predict(
    input_data: csr_matrix, 
    pipeline: Pipeline
) -> np.ndarray:
    """ 
    Predict the model
    Args:
        input_data (csr_matrix): input data
        pipeline (Pipeline): trained model
    Returns:
        np.ndarray: predicted data
    """
    return pipeline.predict(input_data)



@task(name='model_evaluation', tags=['model'])
def model_evaluation(
    y_true : np.ndarray, 
    y_pred : np.ndarray
) -> dict:
    """
    Evaluate the model
    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
    Returns:
        dict: dictionary containing accuracy, precision, recall and f1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



@flow(name='model_training_and_evaluation')
def model_training_and_evaluation(
    X_train,
    y_train,
    X_test,
    y_test,
    **kwargs
) -> dict:
    """
    Train and evaluate the model
    Args:
        X_train (csr_matrix): X train data
        y_train (csr_matrix): y train data
        X_test (csr_matrix): X test data
        y_test (csr_matrix): y test data
        **kwargs: keyword arguments
    Returns:
        dict: dictionary containing accuracy, precision, recall and f1 score
    """

    pipeline = model_training(X_train, y_train, **kwargs)
    y_pred =  model_predict(X_test, pipeline)
    evaluate = model_evaluation(y_test, y_pred)
    return  {'evaluate': evaluate, 'pipeline': pipeline}




@flow(name='Model Training')
def train_model(
    data_path: str, 
    local_storage: Optional[str] = config.OUTPUT_FOLDER,
    save_model: Optional[bool] = True, 
    max_features : Optional[int] = 3000,
    ngram_range : Optional[Tuple[int, int]] = (1, 2),
    use_idf : Optional[bool] = True,
    alpha : Optional[float] = 0.01,
    analyser : Optional[str] = 'word',
    stop_words : Optional[str] = 'english'
) -> None:
    """ 
    Train model and save model and text vectorizer
    Args:
        data_path (str): path to data
        local_storage (str): path to local storage
        save_model (bool): save model
        **kwargs: keyword arguments, passed to preprocess_data and model_training_and_evaluation functions. \
            These are the arguments of the TfidfVectorizer class.
    Returns:
        None
    """

    kwargs = {
        'max_features': max_features,
        'ngram_range': ngram_range,
        'use_idf': use_idf,
        'alpha': alpha,
        'analyser': analyser,
        'stop_words': stop_words
    }

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("Level", "Development")

        extract_data = preprocess_data(data_path, True)

        X = extract_data['X']
        y = extract_data['y']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.log_param("train_set_size", X_train.shape[0])
        mlflow.log_param("test_set_size", X_test.shape[0])

        results = model_training_and_evaluation(X_train, y_train, X_test, y_test, **kwargs)

        mlflow.log_metric("accuracy", results['evaluate']['accuracy'])
        mlflow.log_metric("precision", results['evaluate']['precision'])
        mlflow.log_metric("recall", results['evaluate']['recall'])
        mlflow.log_metric("f1", results['evaluate']['f1'])

        os.makedirs(local_storage, exist_ok=True)
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)

        mlflow.sklearn.log_model(results['pipeline'], "models")
        mlflow.register_model(f"runs:/{run_id}/models", "email_spam_model")

        if save_model:  task_save_pickle(config.MODEL_PATH, results['pipeline'])

        return results


@flow(name='Batch Inference')
def batch_inference(input_path: str) :
    """ Batch inference """
    model = task_load_pickle(config.MODEL_PATH)
    data = preprocess_data(input_path, False)
    return model_predict(data['X'], model)



@flow(name="Data validation")
def data_validation(checkpoint_name: str):
    download_data()
    run_checkpoint_validation(checkpoint_name=checkpoint_name)


mlflow.set_experiment(f'Email Spam')

if __name__ == '__main__':
    data_validation("email_checkpoint")
    train_model(config.DATA_PATH)
    inference = batch_inference(config.DATA_PATH)