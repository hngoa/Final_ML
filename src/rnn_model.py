import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    """
    Base class for recurrent neural network classifiers.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        fc_hidden_size: int = 64,
        output_size: int = 1,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # RNN layer 
        self.rnn = None

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * (2 if bidirectional else 1),
                fc_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        raise NotImplementedError


class LSTMClassifier(BaseRNN):
    """
    LSTM model for Mushroom Classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        fc_hidden_size: int = 64,
        output_size: int = 1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden_size=fc_hidden_size,
            output_size=output_size,
        )

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.embedding_dropout(x)

        _, (hidden, _) = self.rnn(x)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        output = self.classifier(hidden)

        return output


class GRUClassifier(BaseRNN):
    """
    GRU model for Mushroom Classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        fc_hidden_size: int = 64,
        output_size: int = 1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden_size=fc_hidden_size,
            output_size=output_size,
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.embedding_dropout(x)

        _, hidden = self.rnn(x)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        output = self.classifier(hidden)

        return output


def build_lstm(
    vocab_size: int,
    embedding_dim: int = 32,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
):
    return LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


def build_gru(
    vocab_size: int,
    embedding_dim: int = 32,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
):
    return GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )