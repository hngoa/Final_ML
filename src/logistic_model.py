from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_logistic(X_train, y_train, C=1.0, max_iter=200, solver='lbfgs'):
    """
    Huấn luyện mô hình Logistic Regression.

    Args:
        X_train: numpy array (One-Hot encoded features)
        y_train: numpy array (0/1)
        C: Inverse of regularization strength
        max_iter: Số vòng lặp tối đa
        solver: Thuật toán tối ưu ('lbfgs', 'liblinear', 'newton-cg', 'saga')

    Returns:
        model: fitted LogisticRegression
    """
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_logistic(model, X, y):
    """
    Đánh giá mô hình Logistic Regression.

    Returns:
        dict chứa Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    """
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }
