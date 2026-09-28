import pandas as pd
import shap


def explain_malmem_model(model, X):
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        class_index = list(model.classes_).index("Malware")
        values = shap_values[class_index]
    elif shap_values.ndim == 3:
        class_index = list(model.classes_).index("Malware")
        values = shap_values[:, :, class_index]
    else:
        values = shap_values

    return pd.DataFrame(
        values,
        columns=X.columns,
        index=X.index,
    )
