import pandas as pd
import numpy as np

def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Kiểm tra tính toàn vẹn của dữ liệu thô.
    """
    validation_results = {
        "status": "pass",
        "messages": [],
        "shape": df.shape,
        "missing_question_marks": 0,
        "duplicate_rows": 0,
        "constant_columns": []
    }

    # 1. Kiểm tra số cột
    if df.shape[1] != 23:
        validation_results["status"] = "fail"
        validation_results["messages"].append(f"Số lượng cột không hợp lệ: mong đợi 23, hiện tại {df.shape[1]}")

    # 2. Kiểm tra nhãn
    if "class" in df.columns:
        valid_labels = set(df["class"].unique())
        if not valid_labels.issubset({"e", "p"}):
            validation_results["status"] = "fail"
            validation_results["messages"].append(f"Nhãn không hợp lệ. Các nhãn hiện tại: {valid_labels}")
    else:
        validation_results["status"] = "fail"
        validation_results["messages"].append("Không tìm thấy cột 'class' (nhãn).")

    # 3. Kiểm tra dòng trùng lặp
    dup_count = df.duplicated().sum()
    validation_results["duplicate_rows"] = int(dup_count)

    # 4. Kiểm tra ký hiệu '?'
    missing_q = (df == "?").sum().sum()
    validation_results["missing_question_marks"] = int(missing_q)

    # 5. Kiểm tra cột hằng số (chỉ có 1 giá trị duy nhất)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    validation_results["constant_columns"] = constant_cols
    
    if len(validation_results["messages"]) == 0:
        validation_results["messages"].append("Dữ liệu thô hợp lệ.")

    return validation_results
