import numpy as np
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def generate_best_model(model_name: str,
                        X: np.ndarray,
                        y: np.ndarray) -> BaseEstimator:
    """Fetches a model from the model list and runs a grid search
    CV on it. Returns the best model with the best score.

    Models available:
    * SVM: `svm`
    * Naive Bayes: `naive_bayes`
    * Logistic regression: `lr`

    Args:
        model_name (str): name of the model
        X (np.ndarray): The data with features
        y (np.ndarray): The labels

    Returns:
        BaseEstimator: Best version of the model
    """
    model_list = {
        "svm": BestSVM(),
        "naive_bayes": BestNaiveBayes(),
        'lr': BestLogisticRegression()
    }

    model = model_list[model_name]
    return model.fit_best_model(X, y)


class BestModel:
    """Base class for text models
    """
    def __init__(self):
        self.model: BaseEstimator = None
        self.name: str = ""
        self.params: Dict[str, Any] = {}

    def fit_best_model(self,
                       X: np.ndarray,
                       y: np.ndarray) -> Tuple[BaseEstimator, float]:
        gs_cv = GridSearchCV(self.model, self.params, cv=5)
        gs_cv.fit(X, y)

        return gs_cv.best_estimator_, gs_cv.best_score_

    def __str__(self):
        return f"{self.name}\nN° of params: {len(self.params.keys())}"


class BestSVM(BestModel):
    def __init__(self):
        super().__init__()
        self.model = make_pipeline(TfidfVectorizer(), SVC())
        self.name = SVC().__class__.__name__
        self.params = {"svc__C": np.logspace(0, 5, 10),
                       'svc__gamma': np.logspace(-6, 0, 10),
                       'svc__kernel': ["linear", "rbf"]}


class BestNaiveBayes(BestModel):
    def __init__(self):
        super().__init__()
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.name = MultinomialNB().__class__.__name__
        self.params = {"multinomialnb__alpha": np.linspace(0, 1, 20)}


class BestLogisticRegression(BestModel):
    def __init__(self):
        super().__init__()
        self.model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        self.name = MultinomialNB().__class__.__name__
        self.params = {"logisticregression__C": np.logspace(-4, 5, 20),
                       "logisticregression__penalty": ["l1", "l2"]}
