import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def encode_target_ml(y):
    """
    Mã hóa nhãn: e → 0, p → 1.
    """
    return y.map({'e': 0, 'p': 1}).values


def prepare_ml_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Tiền xử lý dữ liệu cho ML truyền thống (Logistic Regression, ...):
    1. One-Hot Encoding: fit trên X_train, transform trên tất cả các tập.
    2. Encode target: e → 0, p → 1.

    Returns:
        X_train_enc, X_val_enc, X_test_enc: numpy arrays (One-Hot encoded)
        y_train_enc, y_val_enc, y_test_enc: numpy arrays (0/1)
        encoder: fitted OneHotEncoder (để tái sử dụng khi dự đoán)
    """
    # 1. One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_enc = encoder.fit_transform(X_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)

    # 2. Encode target
    y_train_enc = encode_target_ml(y_train)
    y_val_enc = encode_target_ml(y_val)
    y_test_enc = encode_target_ml(y_test)

    return X_train_enc, X_val_enc, X_test_enc, y_train_enc, y_val_enc, y_test_enc, encoder
