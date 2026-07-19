import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class MushroomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape for Conv1D: (batch, in_channels=1, sequence_length)
        self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(data_path="data/agaricus-lepiota.data"):
    # Cột đầu tiên là nhãn (Target: e = edible, p = poisonous)
    # Các cột còn lại là features
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", 
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", 
        "stalk-surface-below-ring", "stalk-color-above-ring", 
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", 
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    df = pd.read_csv(data_path, header=None, names=columns)
    
    # Label encoding target (e -> 0, p -> 1)
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])
    
    # Tách Features (X) và Target (y)
    X = df.drop("class", axis=1)
    y = df["class"].values
    
    # One-Hot Encoding cho toàn bộ features
    X_encoded = pd.get_dummies(X)
    
    return X_encoded.values, y

def get_dataloaders(data_path="data/agaricus-lepiota.data", batch_size=64, test_size=0.15, val_size=0.15, random_state=42):
    X, y = load_and_preprocess_data(data_path)
    
    # Chia tập train và tập temp (chứa val + test)
    # Tỉ lệ test_size tương đối trên tổng thể
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )
    
    # Chia tiếp tập temp thành val và test
    val_ratio = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )
    
    # Khởi tạo Datasets
    train_dataset = MushroomDataset(X_train, y_train)
    val_dataset = MushroomDataset(X_val, y_val)
    test_dataset = MushroomDataset(X_test, y_test)
    
    # Khởi tạo DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X.shape[1] # Số lượng features sau khi one-hot encode
    
    return train_loader, val_loader, test_loader, input_dim
