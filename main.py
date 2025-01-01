from sklearn.preprocessing import OneHotEncoder
from data_loader import load_heart_disease_data
from preporcessing import train_test_split, standardize, preprocess_data
from logistic_regression import LogisticRegression
import numpy as np

if __name__ == "__main__":
    # Data load
    X, y = load_heart_disease_data()

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y)

    # Preprocessing
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    X_train, X_test = standardize(X_train, X_test)

    n_classes = y_one_hot.shape[1]
    model = LogisticRegression(learning_rate=0.001, num_iterations=1000, optimizer="sgd")
    model.train(X_train, y_train, n_classes)

    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test_labels) * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")