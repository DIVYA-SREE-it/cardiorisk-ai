import pandas as pd

def predict(model, scaler, input_dict, feature_cols):

    df = pd.DataFrame([input_dict])[feature_cols]

    X = scaler.transform(df)

    risk = model.predict_proba(X)[0][1]

    return risk
