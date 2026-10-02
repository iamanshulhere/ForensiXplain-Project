from pathlib import Path
import csv

from src.data.malmem_data import split_malmem_data
from src.models.malmem_model import train_malmem_model
from src.models.malmem_evaluation import evaluate_malmem_model


def main():
    X_train, X_test, y_train, y_test = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(
        X_train,
        y_train,
        random_state=42,
    )

    metrics = evaluate_malmem_model(
        model,
        X_test,
        y_test,
    )

    output_dir = Path("results/MalMem2022")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "table8_malmem_evaluation.csv"

    cm = metrics["confusion_matrix"]

    rows = [
        [
            "dataset",
            "task",
            "evaluation_type",
            "features",
            "train_samples",
            "test_samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positives",
            "false_negatives",
        ],
        [
            "MalMem2022",
            "Malware classification",
            "Ground-truth-backed supervised evaluation",
            X_train.shape[1],
            X_train.shape[0],
            X_test.shape[0],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            int(cm[0][1]),
            int(cm[1][0]),
        ],
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Created: {output_file}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")


if __name__ == "__main__":
    main()
