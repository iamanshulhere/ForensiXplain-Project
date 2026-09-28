def test_malmem_evaluation_returns_expected_metrics():
    from src.data.malmem_data import split_malmem_data
    from src.models.malmem_model import train_malmem_model
    from src.models.malmem_evaluation import evaluate_malmem_model

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

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1

    assert metrics["confusion_matrix"].shape == (2, 2)
