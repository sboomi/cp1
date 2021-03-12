import numpy as np
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def generate_best_model(model_name: str,
                        X: np.ndarray,
                        y: np.ndarray) -> BaseEstimator:
    model_list = {
        "svm": BestSVM(),
        "naive_bayes": BestNaiveBayes()
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
        self.params = {"multinomialnb__alpha": np.linspace(0, 1, 10)}
