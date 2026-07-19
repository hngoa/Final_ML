import json
import pandas as pd

def build_vocabulary(X_train: pd.DataFrame):
    """
    Xây dựng vocabulary CHỈ từ tập train.
    Token có dạng: column=value (ví dụ: odor=n, cap-shape=x)
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for col in X_train.columns:
        for val in sorted(X_train[col].unique()):
            token = f"{col}={val}"
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def transform_to_tokens(X: pd.DataFrame, vocab: dict):
    """
    Chuyển DataFrame categorical thành ma trận integer tokens.
    Mỗi mẫu có đúng len(X.columns) token (= 21 sau tiền xử lý chung).
    """
    result = []
    for _, row in X.iterrows():
        tokens = []
        for col in X.columns:
            token_str = f"{col}={row[col]}"
            tokens.append(vocab.get(token_str, vocab["<UNK>"]))
        result.append(tokens)
    return result

def save_vocabulary(vocab: dict, path: str):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

def load_vocabulary(path: str):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)
