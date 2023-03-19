import os
import config
from utils import download_data
from helpers import task_load_pickle, task_save_pickle
from preprocessing import preprocess_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix
from prefect import task, flow
from prefect_great_expectations import run_checkpoint_validation
import mlflow
import argparse


@task(name='model_training', tags=['model'])
def model_training(X: csr_matrix, y: csr_matrix) -> MultinomialNB:
    """ Train model """

    model = MultinomialNB()
    model.fit(X, y)
    return model



@task(name='model_predict', tags=['model'])
def model_predict(input_data: csr_matrix, model: MultinomialNB) -> np.ndarray:
    """ Predict model """

    return model.predict(input_data)



@task(name='model_evaluation', tags=['model'])
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



@flow(name='model_training_and_evaluation')
def model_training_and_evaluation(X_train, X_test, y_train, y_test) -> dict:
    """ Train model and evaluate """

    model = model_training(X_train, y_train)
    prediction = model_predict(X_test, model)
    evaluation = model_evaluation(y_test, prediction)
    return {'model': model, 'evaluation': evaluation}




@flow(name='Model Training', retries=1, retry_delay_seconds=30)
def train_model(data_path: str, 
        save_model: bool = True, 
        save_tv: bool = True,
        local_storage: str = config.OUTPUT_FOLDER
) -> None:
    """ Train model and save model and text vectorizer """

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_features', type=int, default=1000)
    parser.add_argument('--lowercase', type=bool, default=True)
    parser.add_argument('--analyzer', type=str, default='word')
    parser.add_argument('--stop_words', type=str, default='english')
    parser.add_argument('--ngram_range', type=tuple, default=(1, 1))
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f'email_spam')

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("Level", "Development")
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("lowercase", args.lowercase)
        mlflow.log_param("analyzer", args.analyzer)
        mlflow.log_param("stop_words", args.stop_words)
        mlflow.log_param("ngram_range", args.ngram_range)


        extract_data = preprocess_data(data_path, None,True,kwargs=vars(args))        
        X = extract_data['X']
        y = extract_data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlflow.log_param("train_set_size", X_train.shape[0])
        mlflow.log_param("test_set_size", X_test.shape[0])

        model = model_training_and_evaluation( X_train, X_test, y_train, y_test)

        mlflow.log_metric("accuracy", model['evaluation']['accuracy'])
        mlflow.log_metric("precision", model['evaluation']['precision'])
        mlflow.log_metric("recall", model['evaluation']['recall'])
        mlflow.log_metric("f1", model['evaluation']['f1'])
    
        os.makedirs(local_storage, exist_ok=True)
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        os.makedirs(config.TV_FOLDER, exist_ok=True)

        mlflow.sklearn.log_model(model['model'], "models")
        mlflow.register_model(f"runs:/{run_id}/models", "BayesModel")

        if save_model:
            task_save_pickle(config.MODEL_PATH,model)
        if save_tv:
            task_save_pickle(config.TV_PATH, extract_data['tv'])


@flow(name='Batch Inference')
def batch_inference(input_path: str, tv=None, model=None) :
    """ Batch inference """

    if tv is None:
        tv = task_load_pickle(config.TV_PATH)
    if model is None:
        model = task_load_pickle(config.MODEL_PATH)['model']

    tv_dict = {'vectorizer': tv}
    data = preprocess_data(input_path,tv_dict , False)
    return model_predict(data['X'], model)


@flow(name="Data validation")
def data_validation(checkpoint_name: str):
    download_data()
    run_checkpoint_validation(checkpoint_name=checkpoint_name)


if __name__ == '__main__':
    data_validation("email_checkpoint")
    train_model(config.DATA_PATH)
    inference = batch_inference(config.DATA_PATH)