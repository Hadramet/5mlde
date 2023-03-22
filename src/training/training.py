import os
import config
import mlflow
import numpy as np

from core import training
from utils import download_data
from helpers import task_load_pickle, task_save_pickle
from preprocessing import preprocess_data

from typing import Optional, Tuple
from scipy.sparse import csr_matrix
from prefect import task, flow
from prefect_great_expectations import run_checkpoint_validation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@task(name='model_training', tags=['model'])
def model_training(X: csr_matrix,y: csr_matrix,**kwargs) -> Pipeline:
    return training.model_training(X, y, **kwargs)

@task(name='model_predict', tags=['model'])
def model_predict( input_data: csr_matrix,pipeline: Pipeline) -> np.ndarray:
    return pipeline.predict(input_data)

@task(name='model_evaluation', tags=['model'])
def model_evaluation(y_true : np.ndarray, y_pred : np.ndarray) -> dict:
    return training.model_evaluation(y_true, y_pred)


@flow(name='Model Training')
def train_model(
    data_path: str, 
    local_storage: Optional[str] = config.OUTPUT_FOLDER,
    save_model: Optional[bool] = True, 
    max_features : Optional[int] = 1000,
    ngram_range : Optional[Tuple[int, int]] = (1, 1),
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
        **kwargs: keyword arguments, passed to preprocess_data and \
            model_training_and_evaluation functions. \
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

        pipeline = model_training(X_train, y_train, **kwargs)
        y_pred =  model_predict(X_test, pipeline)
        results = model_evaluation(y_test, y_pred)

        mlflow.log_metric("accuracy", results['accuracy'])
        mlflow.log_metric("precision", results['precision'])
        mlflow.log_metric("recall", results['recall'])
        mlflow.log_metric("f1", results['f1'])

        os.makedirs(local_storage, exist_ok=True)
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)

        mlflow.sklearn.log_model(pipeline, "models")
        mlflow.register_model(f"runs:/{run_id}/models", "email_spam_model")
        if save_model:  task_save_pickle(config.MODEL_PATH, pipeline)
        return results



@flow(name="Data validation")
def data_validation(checkpoint_name: str):
    download_data()
    run_checkpoint_validation(checkpoint_name=checkpoint_name)


mlflow.set_experiment(f'Email Spam')

if __name__ == '__main__':
    data_validation("email_checkpoint")
    train_model(config.DATA_PATH)