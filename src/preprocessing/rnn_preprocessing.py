import torch
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing.categorical_tokenizer import build_vocabulary, transform_to_tokens


def encode_target(y):
    """
    Encode target labels.

    e -> 0 (Edible)
    p -> 1 (Poisonous)
    """
    return y.map({"e": 0, "p": 1})


def prepare_rnn_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=64,
):
    """
    Prepare DataLoaders for LSTM / GRU.

    Pipeline
    --------
    1. Encode target labels
    2. Convert categorical features to token ids
    3. Convert to PyTorch tensors
    4. Build DataLoaders

    Output shape
    ------------
    X : (batch_size, sequence_length)
    y : (batch_size, 1)
    """
    vocab = build_vocabulary(X_train)

    # =====================================
    # Encode target
    # =====================================

    y_train = encode_target(y_train)
    y_val = encode_target(y_val)
    y_test = encode_target(y_test)

    # =====================================
    # Transform categorical values to tokens
    # =====================================

    X_train_tokens = transform_to_tokens(X_train, vocab)
    X_val_tokens = transform_to_tokens(X_val, vocab)
    X_test_tokens = transform_to_tokens(X_test, vocab)

    # =====================================
    # Metadata
    # =====================================

    metadata = {
        "sequence_length": X_train.shape[1],
        "vocab_size": len(vocab),
    }

    # =====================================
    # Convert to tensors
    # =====================================

    X_train_tensor = torch.tensor(
        X_train_tokens,
        dtype=torch.long,
    )

    X_val_tensor = torch.tensor(
        X_val_tokens,
        dtype=torch.long,
    )

    X_test_tensor = torch.tensor(
        X_test_tokens,
        dtype=torch.long,
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    y_val_tensor = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    # =====================================
    # Dataset
    # =====================================

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor,
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor,
    )

    # =====================================
    # DataLoader
    # =====================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        metadata,
    )