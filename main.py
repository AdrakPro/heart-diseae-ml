from data_loader import load_heart_disease_data
from preporcessing import train_test_split, standardize, preprocess_data
from logistic_regression import LogisticRegression
import numpy as np


if __name__ == "__main__":
    X, y = load_heart_disease_data()

    X = preprocess_data(X)
    # The problem is database is multiclass, where code was binary
    y = np.where(y > 0, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=421
    )

    X_train, X_test = standardize(X_train, X_test)

    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")
