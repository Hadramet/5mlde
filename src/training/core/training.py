import numpy as np
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
