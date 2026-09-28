def test_malmem_shap_returns_feature_contributions():
    from src.data.malmem_data import split_malmem_data
    from src.models.malmem_model import train_malmem_model
    from src.models.malmem_shap import explain_malmem_model

    X_train, X_test, y_train, _ = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(
        X_train,
        y_train,
        random_state=42,
    )

    explanations = explain_malmem_model(
        model,
        X_test.head(10),
    )

    assert explanations.shape[0] == 10
    assert explanations.shape[1] == 55
    assert list(explanations.columns) == list(X_test.columns)
def test_malmem_shap_contains_positive_and_negative_contributions():
    from src.data.malmem_data import split_malmem_data
    from src.models.malmem_model import train_malmem_model
    from src.models.malmem_shap import explain_malmem_model

    X_train, X_test, y_train, _ = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(
        X_train,
        y_train,
        random_state=42,
    )

    explanations = explain_malmem_model(
        model,
        X_test.head(100),
    )

    assert (explanations.to_numpy() > 0).any()
    assert (explanations.to_numpy() < 0).any()
