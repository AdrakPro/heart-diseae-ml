import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from hyperparams_tuning import tune_binary_with_gridsearch, tune_multi
from logistic_regression import BinaryClassificationModel
from multi_logistic_regression import MultiClassificationModel
from preporcessing import Preprocessor
from utils import log_metrics


def main():
    # binary params 0.25 700
    # multi params 0.01 300
    data_path = "data/heart.csv"
    data = pd.read_csv(data_path)

    random_state = 2137
    test_size = 0.33

    # Default params
    hyperparams = {
        "learning_rate": 0.01,
        "num_iterations": 300,
        "momentum": 0.5,
        "beta2": 0.999,
        "epsilon": 1e-7,
    }

    tune_hyperparameters = False
    transform_to_binary_classification = False

    X_train, X_test, y_train, y_test = Preprocessor(data, random_state, test_size).run()

    if transform_to_binary_classification:
        y_train = y_train.apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)
        y_test = y_test.apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values

        if tune_hyperparameters:
            model = tune_binary_with_gridsearch(X_train, y_train)
        else:
            model = BinaryClassificationModel(
                hyperparams["learning_rate"], hyperparams["num_iterations"]
            )

        model.fit(X_train, y_train)

        accuracy, conf_matrix, class_report, logloss = model.evaluate(X_test, y_test)
        log_metrics(
            hyperparams["learning_rate"],
            hyperparams["num_iterations"],
            accuracy,
            conf_matrix,
            class_report,
            logloss,
        )

    else:
        n_classes = len(np.unique(y_train))
        encoder = OneHotEncoder(sparse_output=False)
        y_train = y_train.values.reshape(-1, 1)
        y_train = encoder.fit_transform(y_train)

        if tune_hyperparameters:
            model = tune_multi(X_train, y_train, n_classes)
        else:
            model = MultiClassificationModel(
                n_classes,
                hyperparams["learning_rate"],
                hyperparams["num_iterations"],
                hyperparams["momentum"],
                hyperparams["beta2"],
                hyperparams["epsilon"],
                optimizer="adam",
            )

        model.fit(X_train, y_train)

        accuracy, conf_matrix, class_report, logloss, roc_auc = model.evaluate(
            X_test, y_test
        )
        log_metrics(
            hyperparams["learning_rate"],
            hyperparams["num_iterations"],
            accuracy,
            conf_matrix,
            class_report,
            logloss,
            roc_auc,
        )


if __name__ == "__main__":
    main()
