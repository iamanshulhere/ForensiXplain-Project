from pathlib import Path
import csv

from src.evaluation.generate_table8 import main


def test_table8_malmem_evaluation():
    main()

    output = Path("results/MalMem2022/table8_malmem_evaluation.csv")
    assert output.exists()

    with output.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1

    row = rows[0]

    assert row["dataset"] == "MalMem2022"
    assert row["features"] == "55"
    assert row["train_samples"] == "46883"
    assert row["test_samples"] == "11713"

    assert abs(float(row["accuracy"]) - 0.999915) < 1e-6
    assert abs(float(row["precision"]) - 0.999831) < 1e-6
    assert abs(float(row["recall"]) - 1.0) < 1e-6
    assert abs(float(row["f1"]) - 0.999915) < 1e-6

    assert row["false_positives"] == "1"
    assert row["false_negatives"] == "0"
