import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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

def tune_binary_with_gridsearch(X_train, y_train):



def tune_multi_with_gridsearch(X_train, y_train):
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "num_iterations": [100, 300, 500],
        "momentum": [0.5, 0.7, 0.9],
        "beta2": [0.99, 0.995, 0.999],
        "epsilon": [1e-8],
    }

    model = MultiClassClassifier(
        5,
        param_grid["learning_rate"][0],
        param_grid["num_iterations"][0],
        param_grid["momentum"][0],
        param_grid["beta2"][0],
        param_grid["epsilon"][0],
    )

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=2, scoring="accuracy", n_jobs=-1
    )

    y_train_labels = np.argmax(y_train, axis=1)
    grid_search.fit(X_train, y_train_labels)

    best_model = grid_search.best_estimator_
    print(
        f"Best Multi-Class Model Params: {grid_search.best_params_}, Best Score: {grid_search.best_score_}"
    )

    return best_model
