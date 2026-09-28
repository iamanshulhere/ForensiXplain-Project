def test_malmem_random_forest_trains_and_predicts():
    from src.models.malmem_model import train_malmem_model
    from src.data.malmem_data import split_malmem_data

    X_train, X_test, y_train, _ = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(
        X_train,
        y_train,
        random_state=42,
    )

    predictions = model.predict(X_test)

    assert predictions.shape[0] == X_test.shape[0]
    assert set(predictions).issubset({"Benign", "Malware"})
    assert len(model.feature_importances_) == 55
