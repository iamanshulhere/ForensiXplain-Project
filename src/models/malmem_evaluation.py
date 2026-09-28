from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_malmem_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            pos_label="Malware",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            pos_label="Malware",
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            pos_label="Malware",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=["Benign", "Malware"],
        ),
    }

    return metrics
