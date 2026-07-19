import torch
from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.categorical_tokenizer import build_vocabulary, transform_to_tokens

def encode_target(y):
    """
    Mã hóa nhãn: e → 0, p → 1.
    Thực hiện ở tầng model-specific, KHÔNG ở common preprocessing.
    """
    return y.map({'e': 0, 'p': 1})

def prepare_cnn_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """
    Quy trình tiền xử lý đặc thù cho CNN 1D:
    1. Encode target (e→0, p→1)
    2. Build vocabulary CHỈ từ X_train
    3. Transform tất cả các tập thành token matrix
    4. Tạo DataLoader

    Input shape sau bước này: (batch_size, seq_len=21) — integer tokens
    """
    # 1. Encode target
    y_train_enc = encode_target(y_train)
    y_val_enc = encode_target(y_val)
    y_test_enc = encode_target(y_test)

    # 2. Build vocabulary chỉ từ train
    vocab = build_vocabulary(X_train)

    # 3. Transform to tokens
    X_train_tokens = torch.tensor(transform_to_tokens(X_train, vocab), dtype=torch.long)
    X_val_tokens = torch.tensor(transform_to_tokens(X_val, vocab), dtype=torch.long)
    X_test_tokens = torch.tensor(transform_to_tokens(X_test, vocab), dtype=torch.long)

    y_train_tensor = torch.tensor(y_train_enc.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val_enc.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test_enc.values, dtype=torch.float32).unsqueeze(1)

    # 4. DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_tokens, y_train_tensor),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tokens, y_val_tensor),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tokens, y_test_tensor),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, vocab
