import torch
from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.categorical_tokenizer import transform_to_tokens

def prepare_rnn_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, vocab, batch_size=64):
    """
    Chuẩn bị DataLoader cho GRU/LSTM.
    Input trước khi đưa vào mô hình sẽ có shape (batch_size, sequence_length).
    sequence_length = 21 (số feature).
    """
    # Transform
    X_train_tokens = torch.tensor(transform_to_tokens(X_train, vocab), dtype=torch.long)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_val_tokens = torch.tensor(transform_to_tokens(X_val, vocab), dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    X_test_tokens = torch.tensor(transform_to_tokens(X_test, vocab), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Datasets
    train_dataset = TensorDataset(X_train_tokens, y_train_tensor)
    val_dataset = TensorDataset(X_val_tokens, y_val_tensor)
    test_dataset = TensorDataset(X_test_tokens, y_test_tensor)
    
    # DataLoaders
    # Chỉ train loader sử dụng shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
