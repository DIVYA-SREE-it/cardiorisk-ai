import shap
import numpy as np
import pandas as pd

def run_shap(model, X_sample, feature_names):

    # Choose correct explainer
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_sample)

    shap_values = explainer(X_sample)

    values = shap_values.values

    importance = np.abs(values).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return df
