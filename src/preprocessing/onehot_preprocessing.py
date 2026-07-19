# import joblib
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# def fit_onehot_encoder(X_train: pd.DataFrame):
#     encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     encoder.fit(X_train)
#     return encoder

# def transform_onehot_data(X: pd.DataFrame, encoder: OneHotEncoder):
#     transformed = encoder.transform(X)
#     # Lấy feature names
#     feature_names = encoder.get_feature_names_out(X.columns)
#     return pd.DataFrame(transformed, columns=feature_names, index=X.index)

# def save_onehot_encoder(encoder: OneHotEncoder, path: str):
#     joblib.dump(encoder, path)

# def load_onehot_encoder(path: str):
#     return joblib.load(path)
