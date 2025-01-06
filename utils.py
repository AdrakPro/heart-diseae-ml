import pandas as pd


def fetch_heart_data():
    return pd.read_csv("data/heart.csv")


def log_metrics(
    learning_rate,
    num_iterations,
    accuracy,
    conf_matrix,
    class_report,
    log_loss,
    roc_auc=0,
):
    log_filename = "model_metrics.log"

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print(f"Log Loss: {log_loss:.4f}")
    print(f"Roc auc: {roc_auc:.4f}")

    log_entry = f"[{learning_rate} - {num_iterations}]: \n"
    log_entry += (
        f"Accuracy: {accuracy:.2f}, Log Loss: {log_loss:.4f}\n, Roc: {roc_auc:.4f}"
    )
    log_entry += f"Confusion Matrix:\n{conf_matrix}\n"
    log_entry += f"Classification Report:\n{class_report}\n"
    log_entry += "===========================================\n"

    with open(log_filename, "a") as log_file:
        log_file.write(log_entry)
