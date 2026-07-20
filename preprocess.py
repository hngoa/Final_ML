"""
Script tiền xử lý dữ liệu.
Chạy MỘT LẦN trước khi sử dụng app:
    python preprocess.py
"""
from src.data_processing import (
    load_raw_data, validate_raw_data, clean_common_data,
    separate_features_target, split_dataset,
    build_preprocessing_report, save_processed_data
)
import json


def main():
    print("=" * 50)
    print("  TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 50)

    # 1. Đọc dữ liệu thô
    df_raw = load_raw_data()
    print(f"[1/5] Đọc dữ liệu thô: {df_raw.shape}")

    # 2. Kiểm tra tính toàn vẹn
    val_results = validate_raw_data(df_raw)
    missing_before = val_results["missing_question_marks"]
    print(f"[2/5] Kiểm tra dữ liệu: {val_results['status']} | Missing '?': {missing_before}")

    # 3. Làm sạch
    df_clean, dropped = clean_common_data(df_raw)
    print(f"[3/5] Làm sạch: {df_raw.shape} → {df_clean.shape} | Đã loại cột: {dropped}")

    # 4. Tách features / target & chia tập
    X, y = separate_features_target(df_clean)
    splits, indices = split_dataset(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    print(f"[4/5] Chia dữ liệu: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. Lưu tất cả
    metadata = build_preprocessing_report(df_raw, df_clean, dropped, splits, missing_before)
    output_dir = "data/processed"
    save_processed_data(splits, indices, metadata, output_dir)

    # Lưu thêm df_clean và validation_results
    df_clean.to_csv(f"{output_dir}/df_clean.csv", index=False)
    with open(f"{output_dir}/validation_results.json", "w", encoding='utf-8') as f:
        json.dump(val_results, f, indent=4, ensure_ascii=False)

    print(f"[5/5] Đã lưu vào {output_dir}/")
    print("=" * 50)
    print("✅ Hoàn tất! Có thể chạy app: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
