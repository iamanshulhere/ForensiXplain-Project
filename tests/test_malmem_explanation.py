def test_malmem_explanation_returns_top_features():
    from src.data.malmem_data import split_malmem_data
    from src.models.malmem_model import train_malmem_model
    from src.models.malmem_shap import explain_malmem_model
    from src.models.malmem_explanation import generate_malmem_explanations

    X_train, X_test, y_train, y_test = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    model = train_malmem_model(
        X_train,
        y_train,
        random_state=42,
    )

    X_sample = X_test.head(10)

    shap_values = explain_malmem_model(
        model,
        X_sample,
    )

    explanations = generate_malmem_explanations(
        model,
        X_sample,
        shap_values,
        top_n=5,
    )

    assert explanations.shape[0] == 10
    assert "prediction" in explanations.columns
    assert "top_features" in explanations.columns

    for features in explanations["top_features"]:
        assert len(features) == 5
