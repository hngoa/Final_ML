import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.data_validation import validate_raw_data

def load_raw_data(data_path="data/raw/agaricus-lepiota.data"):
    """
    Đọc dữ liệu thô và gán tên cột.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_path}")
        
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", 
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", 
        "stalk-surface-below-ring", "stalk-color-above-ring", 
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", 
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    df = pd.read_csv(data_path, header=None, names=columns)
    return df

def clean_common_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiền xử lý chung: thay thế giá trị thiếu, loại bỏ cột hằng số.
    KHÔNG encode, KHÔNG tokenization ở đây.
    """
    df_clean = df.copy()
    
    # Xử lý missing values ('?') trong stalk-root
    if 'stalk-root' in df_clean.columns:
        df_clean['stalk-root'] = df_clean['stalk-root'].replace('?', 'missing')
        
    # Loại bỏ các cột hằng số (ví dụ: veil-type)
    constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if constant_cols:
        df_clean = df_clean.drop(columns=constant_cols)
        
    return df_clean, constant_cols

def separate_features_target(df: pd.DataFrame):
    """
    Tách X (features) và y (nhãn gốc dạng chuỗi 'e'/'p').
    KHÔNG encode nhãn ở bước này.
    """
    y = df['class']
    X = df.drop(columns=['class'])
    return X, y

def split_dataset(X: pd.DataFrame, y: pd.Series, test_size=0.15, val_size=0.15, random_state=42):
    """
    Chia dữ liệu thành Train, Validation, Test.
    """
    # Tính toán tỷ lệ tập validation so với tập dữ liệu sau khi đã lấy ra tập test
    val_ratio = val_size / (1.0 - test_size)
    
    # Lấy ra index để chia
    indices = np.arange(len(X))
    
    # Chia train_val và test
    train_val_idx, test_idx, _, y_test = train_test_split(
        indices, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Chia tiếp train và val từ tập train_val
    y_train_val = y.iloc[train_val_idx]
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), (train_idx, val_idx, test_idx)

def build_preprocessing_report(df_raw, df_clean, dropped_cols, splits, missing_before):
    """
    Tạo báo cáo metadata.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples_before": len(df_raw),
        "samples_after": len(df_clean),
        "features_before": df_raw.shape[1] - 1,
        "features_after": df_clean.shape[1] - 1,
        "missing_before": int(missing_before),
        "missing_after": int((df_clean == "?").sum().sum()),
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "constant_columns": dropped_cols,
        "dropped_columns": dropped_cols,
        "missing_strategy": {
            "stalk-root": "separate_category (missing)"
        },
        "label_mapping": {
            "e": 0,
            "p": 1
        },
        "split_ratio": {
            "train": len(X_train) / len(df_raw),
            "validation": len(X_val) / len(df_raw),
            "test": len(X_test) / len(df_raw)
        },
        "split_counts": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test)
        },
        "random_state": 42
    }
    return report

def save_processed_data(splits, indices, metadata, output_dir="data/processed"):
    """
    Lưu trữ dữ liệu trung gian và metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    train_idx, val_idx, test_idx = indices
    
    # Save CSVs
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Save indices
    np.save(f"{output_dir}/train_indices.npy", train_idx)
    np.save(f"{output_dir}/val_indices.npy", val_idx)
    np.save(f"{output_dir}/test_indices.npy", test_idx)
    
    # Save metadata
    with open(f"{output_dir}/preprocessing_metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

def run_common_preprocessing_pipeline():
    """
    Thực thi toàn bộ luồng tiền xử lý chung.
    """
    df_raw = load_raw_data()
    validation_results = validate_raw_data(df_raw)
    
    missing_before = validation_results["missing_question_marks"]
    
    df_clean, dropped_cols = clean_common_data(df_raw)
    X, y = separate_features_target(df_clean)
    
    splits, indices = split_dataset(X, y)
    
    metadata = build_preprocessing_report(df_raw, df_clean, dropped_cols, splits, missing_before)
    save_processed_data(splits, indices, metadata)
    
    return df_raw, df_clean, metadata, validation_results


def load_processed_data(processed_dir="data/processed"):
    """
    Load dữ liệu đã tiền xử lý từ thư mục data/processed.
    Yêu cầu: chạy preprocess.py trước để tạo các file cần thiết.
    """
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_val = pd.read_csv(f"{processed_dir}/X_val.csv")
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").squeeze("columns")
    y_val = pd.read_csv(f"{processed_dir}/y_val.csv").squeeze("columns")
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").squeeze("columns")

    df_clean = pd.read_csv(f"{processed_dir}/df_clean.csv")

    with open(f"{processed_dir}/preprocessing_metadata.json", "r", encoding='utf-8') as f:
        metadata = json.load(f)

    with open(f"{processed_dir}/validation_results.json", "r", encoding='utf-8') as f:
        validation_results = json.load(f)

    splits = (X_train, X_val, X_test, y_train, y_val, y_test)
    return df_clean, metadata, validation_results, splits

