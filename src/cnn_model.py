import torch
import torch.nn as nn

class MushroomCNN1D(nn.Module):
    """
    CNN 1D cho dữ liệu categorical (Mushroom Dataset).

    Kiến trúc:
        Input (batch, seq_len=21) — integer tokens
        → Embedding(vocab_size, embed_dim) → (batch, 21, embed_dim)
        → Permute → (batch, embed_dim, 21)
        → Conv1d + ReLU
        → BatchNorm1d (tùy chọn)
        → MaxPool1d
        → Flatten
        → Dropout → FC1 → ReLU
        → Dropout → FC2 → Sigmoid
    """

    def __init__(self, vocab_size, embed_dim=16, num_filters=32,
                 kernel_size=3, dropout_rate=0.3, use_batch_norm=True,
                 seq_len=21):
        super(MushroomCNN1D, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # <PAD> = 0
        )

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(num_filters)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # Kích thước sau Conv1d (padding=same) + MaxPool1d(2)
        pooled_len = seq_len // 2
        self.flatten_dim = num_filters * pooled_len

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len=21) — integer tokens
        x = self.embedding(x)          # → (batch, 21, embed_dim)
        x = x.permute(0, 2, 1)         # → (batch, embed_dim, 21) — channels first

        x = self.conv1(x)              # → (batch, num_filters, 21)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)               # → (batch, num_filters, 10)

        x = torch.flatten(x, 1)        # → (batch, num_filters * 10)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
