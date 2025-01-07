import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from logistic_regression import BinaryClassificationModel
from multi_logistic_regression import MultiClassificationModel


class BinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, num_iterations=300):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model = None

    def fit(self, X, y):
        self.model = BinaryClassificationModel(self.learning_rate, self.num_iterations)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        return np.mean(y_pred == y_true)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)


class MultiClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_classes,
        learning_rate,
        num_iterations,
        momentum,
        beta2,
        epsilon,
    ):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.beta2 = beta2
        self.epsilon = epsilon
        self.model = None

    def fit(self, X, y):
        self.model = MultiClassificationModel(
            self.n_classes,
            self.learning_rate,
            self.num_iterations,
            self.momentum,
            self.beta2,
            self.epsilon,
            optimizer="adam",
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        return np.mean(y_pred == y_true)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def tune_binary_with_gridsearch(X_train, y_train, batch_size):
    param_grid = {
        "learning_rate": [0.1, 0.25, 0.01, 0.05, 0.001, 0.3, 0.2, 0.24],
        "num_iterations": (200, 300, 400, 500, 600, 700, 800, 900, 1000),
    }

    model = BinaryClassifier()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train, batch_size)

    best_model = grid_search.best_estimator_
    print(
        f"Best Binary Model Params: {grid_search.best_params_}, Best Score: {grid_search.best_score_}"
    )

    return best_model


# Custom accuracy scorer
def accuracy_scorer(y_true, y_pred):
    # Convert one-hot encoded to class indices
    y_true = y_true.argmax(axis=1)
    return accuracy_score(y_true, y_pred)


def roc_auc_direct_scorer(estimator, X, y):
    y_proba = estimator.predict_proba(X)
    return roc_auc_score(y, y_proba, multi_class="ovr")


def tune_multi(X_train, y_train, n_classes, batch_size):
    param_grid = {
        "learning_rate": [0.1, 0.25, 0.01, 0.001, 0.2, 0.3],
        "num_iterations": [200, 300, 400, 500, 600, 700],
        "momentum": [0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95],
        "beta2": [0.97, 0.98, 0.99, 0.999],
        "epsilon": [1e-8, 1e-7, 1e-6],
    }

    model = MultiClassClassifier(
        n_classes,
        param_grid["learning_rate"][0],
        param_grid["num_iterations"][0],
        param_grid["momentum"][0],
        param_grid["beta2"][0],
        param_grid["epsilon"][0],
    )

    # custom_scorer = make_scorer(accuracy_scorer, greater_is_better=True)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=roc_auc_direct_scorer,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train, batch_size)

    best_model = grid_search.best_estimator_
    print(
        f"Best Multi-Class Model Params: {grid_search.best_params_}, Best Score: {grid_search.best_score_}"
    )

    return best_model
