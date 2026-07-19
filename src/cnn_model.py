import torch
import torch.nn as nn

class MushroomCNN1D(nn.Module):
    def __init__(self, input_dim, num_filters=16, kernel_size=3, dropout_rate=0.2, use_batch_norm=False):
        super(MushroomCNN1D, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Layer tích chập 1D
        # input shape: (batch_size, in_channels=1, sequence_length=input_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(num_filters)
            
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Tính kích thước sau khi qua Conv1d và MaxPool1d
        # Padding = kernel_size // 2 nên độ dài chuỗi gần như không đổi sau Conv1d.
        # Sau khi MaxPool1d(2), độ dài chuỗi giảm đi một nửa.
        self.flatten_dim = num_filters * (input_dim // 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Lớp Fully Connected
        self.fc1 = nn.Linear(self.flatten_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, 1, input_dim)
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
