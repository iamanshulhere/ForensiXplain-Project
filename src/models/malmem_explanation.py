import pandas as pd


def generate_malmem_explanations(
    model,
    X,
    shap_values,
    top_n=5,
):
    predictions = model.predict(X)

    rows = []

    for row_index, prediction in enumerate(predictions):
        row_shap = shap_values.iloc[row_index]

        ranked_features = (
            row_shap.abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        top_features = [
            {
                "feature": feature,
                "shap_value": float(row_shap[feature]),
                "feature_value": X.iloc[row_index][feature],
            }
            for feature in ranked_features
        ]

        rows.append(
            {
                "prediction": prediction,
                "top_features": top_features,
            }
        )

    return pd.DataFrame(rows)
