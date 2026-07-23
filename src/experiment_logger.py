import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_average_metrics(history):
    return {
        "train_loss": np.mean(history["train_loss"]),
        "val_loss": np.mean(history["val_loss"]),
        "train_acc": np.mean(history["train_acc"]),
        "val_acc": np.mean(history["val_acc"]),
        "train_precision": np.mean(history["train_precision"]),
        "val_precision": np.mean(history["val_precision"]),
        "train_recall": np.mean(history["train_recall"]),
        "val_recall": np.mean(history["val_recall"]),
        "train_f1": np.mean(history["train_f1"]),
        "val_f1": np.mean(history["val_f1"]),
    }


def compute_best_metrics(history):
    idx = int(np.argmin(history["val_loss"]))

    return idx + 1, {
        "train_loss": history["train_loss"][idx],
        "val_loss": history["val_loss"][idx],
        "train_acc": history["train_acc"][idx],
        "val_acc": history["val_acc"][idx],
        "train_precision": history["train_precision"][idx],
        "val_precision": history["val_precision"][idx],
        "train_recall": history["train_recall"][idx],
        "val_recall": history["val_recall"][idx],
        "train_f1": history["train_f1"][idx],
        "val_f1": history["val_f1"][idx],
    }


def compute_worst_metrics(history):
    idx = int(np.argmax(history["val_loss"]))

    return idx + 1, {
        "train_loss": history["train_loss"][idx],
        "val_loss": history["val_loss"][idx],
        "train_acc": history["train_acc"][idx],
        "val_acc": history["val_acc"][idx],
        "train_precision": history["train_precision"][idx],
        "val_precision": history["val_precision"][idx],
        "train_recall": history["train_recall"][idx],
        "val_recall": history["val_recall"][idx],
        "train_f1": history["train_f1"][idx],
        "val_f1": history["val_f1"][idx],
    }


def save_experiment_log(
    model_name,
    run_name,
    history,
    test_metrics,
    hyperparameters,
    train_time,
    test_time,
    train_size,
    val_size,
    test_size,
    device,
    model,
):
    os.makedirs("experiments", exist_ok=True)

    csv_path = "experiments/experiments_log.csv"

    file_exists = os.path.isfile(csv_path)

    best_epoch, best = compute_best_metrics(history)
    worst_epoch, worst = compute_worst_metrics(history)
    avg = compute_average_metrics(history)

    avg_epoch_time = train_time / len(history["train_loss"])

    acc_gap = abs(best["train_acc"] - best["val_acc"])
    precision_gap = abs(best["train_precision"] - best["val_precision"])
    recall_gap = abs(best["train_recall"] - best["val_recall"])
    f1_gap = abs(best["train_f1"] - best["val_f1"])

    header = [
        "Date Time",
        "Model",
        "Run Name",
        "Device",

        "Train Samples",
        "Validation Samples",
        "Test Samples",

        "Hyperparameters",

        "Training Time (s)",
        "Testing Time (s)",
        "Average Epoch Time (s)",

        "Parameters",

        "Best Epoch",

        "Best Train Loss",
        "Best Validation Loss",

        "Best Train Accuracy",
        "Best Validation Accuracy",

        "Best Train Precision",
        "Best Validation Precision",

        "Best Train Recall",
        "Best Validation Recall",

        "Best Train F1",
        "Best Validation F1",

        "Worst Epoch",

        "Worst Train Loss",
        "Worst Validation Loss",

        "Worst Train Accuracy",
        "Worst Validation Accuracy",

        "Worst Train Precision",
        "Worst Validation Precision",

        "Worst Train Recall",
        "Worst Validation Recall",

        "Worst Train F1",
        "Worst Validation F1",

        "Average Train Loss",
        "Average Validation Loss",

        "Average Train Accuracy",
        "Average Validation Accuracy",

        "Average Train Precision",
        "Average Validation Precision",

        "Average Train Recall",
        "Average Validation Recall",

        "Average Train F1",
        "Average Validation F1",

        "Test Loss",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",

        "Accuracy Gap",
        "Precision Gap",
        "Recall Gap",
        "F1 Gap",

        "Confusion Matrix",
        "Classification Report",
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name,
        run_name,
        device,

        train_size,
        val_size,
        test_size,

        hyperparameters,

        round(train_time, 4),
        round(test_time, 4),
        round(avg_epoch_time, 4),

        count_parameters(model),

        best_epoch,

        best["train_loss"],
        best["val_loss"],

        best["train_acc"],
        best["val_acc"],

        best["train_precision"],
        best["val_precision"],

        best["train_recall"],
        best["val_recall"],

        best["train_f1"],
        best["val_f1"],

        worst_epoch,

        worst["train_loss"],
        worst["val_loss"],

        worst["train_acc"],
        worst["val_acc"],

        worst["train_precision"],
        worst["val_precision"],

        worst["train_recall"],
        worst["val_recall"],

        worst["train_f1"],
        worst["val_f1"],

        avg["train_loss"],
        avg["val_loss"],

        avg["train_acc"],
        avg["val_acc"],

        avg["train_precision"],
        avg["val_precision"],

        avg["train_recall"],
        avg["val_recall"],

        avg["train_f1"],
        avg["val_f1"],

        test_metrics["Loss"],
        test_metrics["Accuracy"],
        test_metrics["Precision"],
        test_metrics["Recall"],
        test_metrics["F1-Score"],

        acc_gap,
        precision_gap,
        recall_gap,
        f1_gap,

        test_metrics["Confusion Matrix"].tolist(),
        test_metrics["Classification Report"],
    ]

    # ----------------------------------------------------
    # Kiểm tra Best Model TRƯỚC khi ghi log
    # ----------------------------------------------------
    global_best_acc = 0.0

    if file_exists:
        try:
            df = pd.read_csv(csv_path)

            model_logs = df[df["Model"] == model_name]

            if not model_logs.empty:
                global_best_acc = model_logs["Test Accuracy"].max()

        except Exception:
            pass

    is_new_best = test_metrics["Accuracy"] > global_best_acc

    # ----------------------------------------------------
    # Ghi log
    # ----------------------------------------------------
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(row)

    return is_new_best, global_best_acc

def save_logistic_log(
    run_name,
    hyperparameters,
    metrics,
    train_time,
    test_time,
):
    os.makedirs("experiments", exist_ok=True)

    csv_path = "experiments/experiments_log.csv"

    hp_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())

    row = {
        "Date Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "Logistic Regression",
        "Run Name": run_name,

        "Device": "",
        "Training Time (s)": round(train_time, 4),
        "Testing Time (s)": round(test_time, 4),

        "Parameters": "",
        "Train Samples": "",
        "Validation Samples": "",
        "Test Samples": "",

        "Hyperparameters": hp_str,

        "Best Epoch": "",
        "Worst Epoch": "",

        "Best Train Loss": "",
        "Best Validation Loss": "",
        "Best Train Accuracy": "",
        "Best Validation Accuracy": "",
        "Best Train Precision": "",
        "Best Validation Precision": "",
        "Best Train Recall": "",
        "Best Validation Recall": "",
        "Best Train F1": "",
        "Best Validation F1": "",

        "Worst Train Loss": "",
        "Worst Validation Loss": "",
        "Worst Train Accuracy": "",
        "Worst Validation Accuracy": "",
        "Worst Train Precision": "",
        "Worst Validation Precision": "",
        "Worst Train Recall": "",
        "Worst Validation Recall": "",
        "Worst Train F1": "",
        "Worst Validation F1": "",

        "Average Train Loss": "",
        "Average Validation Loss": "",
        "Average Train Accuracy": "",
        "Average Validation Accuracy": "",
        "Average Train Precision": "",
        "Average Validation Precision": "",
        "Average Train Recall": "",
        "Average Validation Recall": "",
        "Average Train F1": "",
        "Average Validation F1": "",

        "Test Accuracy": round(metrics["Accuracy"], 4),
        "Test Precision": round(metrics["Precision"], 4),
        "Test Recall": round(metrics["Recall"], 4),
        "Test F1": round(metrics["F1-Score"], 4),

        "Accuracy Gap": "",
        "Precision Gap": "",
        "Recall Gap": "",
        "F1 Gap": "",

        "Confusion Matrix": str(metrics["Confusion Matrix"].tolist()),
        "Classification Report": "",
    }

    # Đọc header hiện có
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in header])