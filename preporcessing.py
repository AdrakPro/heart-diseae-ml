import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, data, random_state=2137, test_size=0.33):
        # Remove outliners
        self.data = data[data["trestbps"] != 0]

        self.random_state = random_state
        self.test_size = test_size

    def split_data(self):
        X = self.data.drop("num", axis=1)
        y = self.data["num"]

        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def define_transformers(self, X_train):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline(
            [
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return preprocessor

    def run(self):
        X_train, X_test, y_train, y_test = self.split_data()

        # X_train, y_train = self.handle_outliers(X_train, y_train)

        preprocessor = self.define_transformers(X_train)

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        return X_train_processed, X_test_processed, y_train, y_test
