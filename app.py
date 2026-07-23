import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import json
from datetime import datetime
import ast
matplotlib.rcParams['font.size'] = 11

from src.data_processing import (
    load_raw_data,
    load_processed_data
)


# ── Màu sắc trung tính xuyên suốt ──
COLOR_EDIBLE = "#4b8bbe"
COLOR_POISONOUS = "#c4574b"
COLOR_NEUTRAL = "#6b7280"
COLOR_BAR = "#7c8db5"

BAR_COLORS = [
    "#dbe4f3",   
    "#b9c6e3",   
    "#7c8db5",   
    "#4f618d"   
]

# ── Bảng viết tắt đặc trưng ──
FEATURE_ABBREVIATIONS = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted", "missing": "missing"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"},
}

def load_css(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# ═══════════════════════════════════════════════
#  CÁC HÀM VẼ BIỂU ĐỒ
# ═══════════════════════════════════════════════

def render_kpi_cards(meta):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng mẫu", f"{meta['samples_after']:,}")
    c2.metric("Đặc trưng", f"{meta['features_after']}")
    c3.metric("Ký hiệu '?'", "0", delta=f"-{meta['missing_before']}", delta_color="inverse")
    c4.metric("Dòng trùng", f"{meta['duplicate_rows']}")
    c5.metric("Phân loại", "Nhị phân")


def plot_class_distribution(df):
    counts = df['class'].value_counts()
    labels = [f"Edible ({counts.get('e',0)})", f"Poisonous ({counts.get('p',0)})"]
    sizes = [counts.get('e', 0), counts.get('p', 0)]
    colors = [COLOR_EDIBLE, COLOR_POISONOUS]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title("Phân bố nhãn", fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    return fig


def plot_missing_values(df_raw):
    missing = (df_raw == '?').sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        st.success("Không có giá trị '?' trong dữ liệu gốc.")
        return None

    fig, ax = plt.subplots(figsize=(6, max(2, len(missing) * 0.6)))
    ax.barh(missing.index, missing.values, color=COLOR_POISONOUS, height=0.5, edgecolor='white')
    ax.set_xlabel("Số lượng")
    ax.set_title("Thuộc tính có giá trị khuyết thiếu ('?')", fontweight='bold')
    for i, v in enumerate(missing.values):
        pct = v / len(df_raw) * 100
        ax.text(v + 20, i, f"{v} ({pct:.1f}%)", va='center', fontsize=9, color=COLOR_NEUTRAL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_cardinality(df_raw):
    card = df_raw.drop(columns=['class']).nunique().sort_values(ascending=True)
    colors = [COLOR_POISONOUS if v <= 1 else COLOR_BAR for v in card.values]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(card.index, card.values, color=colors, height=0.6, edgecolor='white')
    ax.set_xlabel("Số giá trị duy nhất")
    ax.set_title("Cardinality của từng đặc trưng", fontweight='bold')
    for i, v in enumerate(card.values):
        ax.text(v + 0.15, i, str(v), va='center', fontsize=8.5, color=COLOR_NEUTRAL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_by_target(df_raw, feature):
    ct = pd.crosstab(df_raw[feature], df_raw['class'], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind='bar', stacked=True, color=[COLOR_EDIBLE, COLOR_POISONOUS], ax=ax, width=0.7, edgecolor='white')
    ax.set_title(f"Tỷ lệ Edible / Poisonous theo '{feature}'", fontweight='bold')
    ax.set_ylabel("Tỷ lệ %")
    ax.legend(title='Class', labels=['Edible', 'Poisonous'], loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return fig


def plot_split_distribution(y_train, y_val, y_test):
    sets = {'Train': y_train, 'Validation': y_val, 'Test': y_test}
    labels = list(sets.keys())
    edible = [int((s == 'e').sum()) for s in sets.values()]
    poisonous = [int((s == 'p').sum()) for s in sets.values()]

    x = np.arange(len(labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - w/2, edible, w, label='Edible (e)', color=COLOR_EDIBLE, edgecolor='white')
    ax.bar(x + w/2, poisonous, w, label='Poisonous (p)', color=COLOR_POISONOUS, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân bố nhãn trong các tập dữ liệu", fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig





def render_abbreviation_table(feature_name):
    if feature_name in FEATURE_ABBREVIATIONS:
        abbr = FEATURE_ABBREVIATIONS[feature_name]
        df_abbr = pd.DataFrame({
            "Ký hiệu": list(abbr.keys()),
            "Ý nghĩa": list(abbr.values())
        })
        st.table(df_abbr)
    else:
        st.caption("Không có danh mục viết tắt cho đặc trưng này.")



# ═══════════════════════════════════════════════
#  KHỞI TẠO ỨNG DỤNG
# ═══════════════════════════════════════════════

st.set_page_config(page_title="ML Dashboard – Mushroom", layout="wide", page_icon="🍄")
load_css("assets/style.css")

@st.cache_data
def load_all_data():
    processed_dir = "data/processed"
    if not os.path.exists(f"{processed_dir}/X_train.csv"):
        raise FileNotFoundError(
            "Chưa có dữ liệu đã xử lý. Vui lòng chạy: python preprocess.py"
        )
    df_raw = load_raw_data()
    df_clean, metadata, validation_results, splits = load_processed_data()
    return df_raw, df_clean, metadata, validation_results, splits

try:
    df_raw, df_clean, metadata, validation_results, splits = load_all_data()
    X_train, X_val, X_test, y_train, y_val, y_test = splits
except Exception as e:
    st.error(f"Lỗi khi xử lý dữ liệu: {e}")
    st.stop()

# ── SIDEBAR ──
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)
else:
    st.sidebar.title("🍄 Mushroom")

if 'page' not in st.session_state:
    st.session_state['page'] = "📄 Dashboard EDA"

def set_page(page_name):
    st.session_state['page'] = page_name

st.sidebar.markdown("### Menu")

if st.sidebar.button("📄 Dashboard EDA", use_container_width=True):
    set_page("📄 Dashboard EDA")

if st.sidebar.button("⚙️ Huấn luyện", use_container_width=True):
    set_page("⚙️ Huấn luyện")

if st.sidebar.button("📊 Đánh giá", use_container_width=True):
    set_page("📊 Đánh giá")

page = st.session_state['page']

# ═══════════════════════════════════════════════
#  TRANG 1: DASHBOARD EDA
# ═══════════════════════════════════════════════

if page == "📄 Dashboard EDA":
    st.title("Phân tích Dữ liệu & Tiền xử lý")

    # ── 1. KPI ──
    st.header("1 · Tổng quan")
    render_kpi_cards(metadata)
    st.markdown("---")

    # ── 2 & 3. Class Distribution + Missing ──
    left, right = st.columns(2)
    with left:
        st.header("2 · Phân bố nhãn")
        st.pyplot(plot_class_distribution(df_raw))


    with right:
        st.header("3 · Giá trị khuyết thiếu")
        fig_miss = plot_missing_values(df_raw)
        if fig_miss:
            st.pyplot(fig_miss)
        miss_df = pd.DataFrame({
            "Thuộc tính": ["stalk-root"],
            "Số lượng": [metadata['missing_before']],
            "Tỷ lệ": [f"{metadata['missing_before']/metadata['samples_before']*100:.1f}%"],
            "Xử lý": ["→ 'missing'"]
        })
        st.table(miss_df)
    st.markdown("---")

    # ── 4. Cardinality ──
    st.header("4 · Cardinality")
    st.pyplot(plot_feature_cardinality(df_raw))

    st.markdown("---")

    # ── 5. Feature vs Target + Abbreviation Table ──
    st.header("5 · Phân bố thuộc tính theo nhãn")
    features = [c for c in df_clean.columns if c != 'class']
    sel = st.selectbox("Chọn đặc trưng:", features, index=features.index('odor') if 'odor' in features else 0)

    col_chart, col_abbr = st.columns([3, 2])
    with col_chart:
        st.pyplot(plot_feature_by_target(df_raw, sel))
    with col_abbr:
        st.subheader("Danh mục viết tắt")
        render_abbreviation_table(sel)
    st.markdown("---")

    # ── 6. Train / Val / Test ──
    st.header("6 · Phân chia tập dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train (70%)", f"{metadata['split_counts']['train']:,}")
    c2.metric("Validation (15%)", f"{metadata['split_counts']['validation']:,}")
    c3.metric("Test (15%)", f"{metadata['split_counts']['test']:,}")
    st.pyplot(plot_split_distribution(y_train, y_val, y_test))
    st.markdown("---")


# ═══════════════════════════════════════════════
#  TRANG 2: HUẤN LUYỆN
# ═══════════════════════════════════════════════

elif page == "⚙️ Huấn luyện":

    st.title("⚙️ Huấn luyện Mô hình")

    # ── Chọn mô hình ──
    model_type = st.selectbox(
        "Chọn mô hình",
        [
            "CNN 1D",
            "LSTM",
            "GRU",
            "Logistic Regression",
        ]
    )

    # ── Sidebar chung ──
    st.sidebar.header("Tên thực nghiệm")

    if "exp_name" not in st.session_state:
        st.session_state.exp_name = datetime.now().strftime("Run_%H_%M_%S")

    exp_name = st.sidebar.text_input(
        "Experiment Name",
        key="exp_name"
    )

    # ══════════════════════════════════════════
    #  CNN 1D
    # ══════════════════════════════════════════
    if model_type == "CNN 1D":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import time

        from src.preprocessing.cnn_preprocessing import prepare_cnn_dataloaders
        from src.cnn_model import MushroomCNN1D
        from src.engine import train_step, eval_step, evaluate_metrics
        from src.experiment_logger import save_experiment_log

        st.sidebar.header("Siêu tham số")
        batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
        epochs = st.sidebar.slider("Epochs", 5, 100, 30, step=5)
        learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)

        st.sidebar.header("Kiến trúc CNN")
        embed_dim = st.sidebar.selectbox("Embedding Dim", [8, 16, 32], index=1)
        num_filters = st.sidebar.selectbox("Số Filters", [16, 32, 64], index=1)
        kernel_size = st.sidebar.selectbox("Kernel Size", [3, 5, 7], index=0)
        dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3, step=0.05)
        use_batch_norm = st.sidebar.checkbox("Batch Normalization", value=True)

        # ── Hiển thị kiến trúc ──
        st.subheader("Kiến trúc mô hình")
        arch_text = f"""
| Lớp | Chi tiết |
|---|---|
| **Embedding** | vocab_size → {embed_dim}d |
| **Conv1d** | in={embed_dim}, out={num_filters}, kernel={kernel_size}, padding={kernel_size//2} |
| **BatchNorm1d** | {'Có' if use_batch_norm else 'Không'} |
| **Activation** | ReLU |
| **MaxPool1d** | kernel=2 |
| **Dropout** | {dropout_rate} |
| **FC1** | {num_filters * (21 // 2)} → 64, ReLU |
| **FC2** | 64 → 1, Sigmoid |
| **Loss** | BCELoss |
| **Optimizer** | Adam (lr={learning_rate}) |
"""
        st.markdown(arch_text)

        # ── Nút Huấn luyện ──
        if st.button("🚀 Bắt đầu Huấn luyện", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Loss")
                loss_chart = st.empty()
            with col_r:
                st.subheader("Accuracy")
                acc_chart = st.empty()

            # 1. Chuẩn bị dữ liệu
            status.text("Đang tokenize dữ liệu...")
            train_loader, val_loader, test_loader, vocab = prepare_cnn_dataloaders(
                X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
            )

            # 2. Khởi tạo mô hình
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MushroomCNN1D(
                vocab_size=len(vocab),
                embed_dim=embed_dim,
                num_filters=num_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                seq_len=21
            ).to(device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 3. Vòng lặp huấn luyện
            history = {
                "train_loss": [],
                "val_loss": [],

                "train_acc": [],
                "val_acc": [],

                "train_precision": [],
                "val_precision": [],

                "train_recall": [],
                "val_recall": [],

                "train_f1": [],
                "val_f1": [],
            }

            train_start = time.perf_counter()
            best_val_loss = float('inf')
            os.makedirs("models", exist_ok=True)

            st.session_state['cnn_config'] = {
                'vocab_size': len(vocab), 'embed_dim': embed_dim,
                'num_filters': num_filters, 'kernel_size': kernel_size,
                'dropout_rate': dropout_rate, 'use_batch_norm': use_batch_norm
            }
            st.session_state['vocab'] = vocab

            status.text(f"Huấn luyện trên: {device}")

            for epoch in range(epochs):
                t_loss, t_acc, t_pre, t_rec, t_f1 = train_step(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )

                v_loss, v_acc, v_pre, v_rec, v_f1, _, _ = eval_step(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                history["train_loss"].append(t_loss)
                history["val_loss"].append(v_loss)

                history["train_acc"].append(t_acc)
                history["val_acc"].append(v_acc)

                history["train_precision"].append(t_pre)
                history["val_precision"].append(v_pre)

                history["train_recall"].append(t_rec)
                history["val_recall"].append(v_rec)

                history["train_f1"].append(t_f1)
                history["val_f1"].append(v_f1)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    torch.save(model.state_dict(), "models/temp_best_cnn_model.pth")

                loss_chart.line_chart(pd.DataFrame({
                    'Train Loss': history['train_loss'],
                    'Val Loss': history['val_loss']
                }))
                acc_chart.line_chart(pd.DataFrame({
                    'Train Acc': history['train_acc'],
                    'Val Acc': history['val_acc']
                }))

                progress.progress((epoch + 1) / epochs)
                status.text(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
                time.sleep(0.02)

            train_end = time.perf_counter()
            st.success(f"✅ Hoàn tất! Best Val Loss: {best_val_loss:.4f}. Đang đánh giá trên tập Test...")

            test_start = time.perf_counter()

            model.load_state_dict(torch.load("models/temp_best_cnn_model.pth"))
            model.eval()

            test_loss, test_acc, _, _, _, preds, targets = eval_step(
                model,
                test_loader,
                criterion,
                device,
            )

            test_end = time.perf_counter()

            metrics = evaluate_metrics(targets, preds)
            metrics["Loss"] = test_loss

            run_name = st.session_state.exp_name.strip()

            if not run_name:
                run_name = datetime.now().strftime("Run_%H_%M_%S")
                st.session_state.exp_name = run_name

            hp_str = (
                f"batch={batch_size}, "
                f"epochs={epochs}, "
                f"lr={learning_rate}, "
                f"embed={embed_dim}, "
                f"filters={num_filters}, "
                f"kernel={kernel_size}, "
                f"dropout={dropout_rate}, "
                f"bn={use_batch_norm}"
            )
            
            is_new_best, global_best_acc = save_experiment_log(
                model_name="CNN",
                run_name=run_name,
                history=history,
                test_metrics=metrics,
                hyperparameters=hp_str,
                train_time=train_end - train_start,
                test_time=test_end - test_start,
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                test_size=len(test_loader.dataset),
                device=str(device),
                model=model,
            )

            if is_new_best:
                if os.path.exists("models/temp_best_cnn_model.pth"):
                    os.replace("models/temp_best_cnn_model.pth", "models/best_cnn_model.pth")
                with open("models/best_cnn_config.json", "w") as f:
                    json.dump(st.session_state['cnn_config'], f)
                st.success(f"🏆 Cấu hình này đã đạt Test Accuracy tốt nhất ({metrics['Accuracy']:.4f}) và được lưu làm Best Model!")
            else:
                if os.path.exists("models/temp_best_cnn_model.pth"):
                    os.remove("models/temp_best_cnn_model.pth")
                st.info(f"💾 Test Accuracy {metrics['Accuracy']:.4f} (Kỷ lục CNN: {global_best_acc:.4f}). Kết quả đã lưu lịch sử.")

    # ══════════════════════════════════════════
    #  LOGISTIC REGRESSION
    # ══════════════════════════════════════════
    elif model_type == "Logistic Regression":
        import time
        from src.preprocessing.onehot_preprocessing import prepare_ml_data
        from src.logistic_model import train_logistic, evaluate_logistic
        from src.experiment_logger import save_logistic_log

        st.sidebar.header("Siêu tham số")
        C_val = st.sidebar.selectbox("C (Regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], index=2)
        max_iter = st.sidebar.selectbox("Max Iterations", [100, 200, 500, 1000], index=1)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "saga"], index=0)

        # ── Hiển thị kiến trúc ──
        st.subheader("Kiến trúc mô hình")
        arch_text = f"""
| Thành phần | Chi tiết |
|---|---|
| **Tiền xử lý** | One-Hot Encoding |
| **Mô hình** | Logistic Regression (sklearn) |
| **C** | {C_val} |
| **Max Iterations** | {max_iter} |
| **Solver** | {solver} |
| **Random State** | 42 |
"""
        st.markdown(arch_text)

        # ── Nút Huấn luyện ──
        if st.button("🚀 Bắt đầu Huấn luyện", use_container_width=True):
            status = st.empty()

            # 1. Chuẩn bị dữ liệu
            status.text("Đang One-Hot Encoding dữ liệu...")
            X_tr_enc, X_val_enc, X_te_enc, y_tr_enc, y_val_enc, y_te_enc, encoder = prepare_ml_data(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            st.info(f"Số đặc trưng sau One-Hot Encoding: {X_tr_enc.shape[1]}")

            # 2. Huấn luyện
            status.text("Đang huấn luyện Logistic Regression...")

            start_train = time.perf_counter()

            lr_model = train_logistic(
                X_tr_enc,
                y_tr_enc,
                C=C_val,
                max_iter=max_iter,
                solver=solver
            )

            train_time = time.perf_counter() - start_train

            # 3. Đánh giá
            start_test = time.perf_counter()

            metrics_train = evaluate_logistic(lr_model, X_tr_enc, y_tr_enc)
            metrics_val = evaluate_logistic(lr_model, X_val_enc, y_val_enc)
            metrics_test = evaluate_logistic(lr_model, X_te_enc, y_te_enc)

            test_time = time.perf_counter() - start_test

            st.success("✅ Hoàn tất huấn luyện!")

            # Hiển thị kết quả 3 tập
            col1, col2, col3 = st.columns(3)
            col1.metric("Train Accuracy", f"{metrics_train['Accuracy']:.4f}")
            col2.metric("Val Accuracy", f"{metrics_val['Accuracy']:.4f}")
            col3.metric("Test Accuracy", f"{metrics_test['Accuracy']:.4f}")

            # 4. Ghi log
            run_name = st.session_state.exp_name.strip()

            if not run_name:
                run_name = datetime.now().strftime("Run_%H_%M_%S")
                st.session_state.exp_name = run_name

            save_logistic_log(
                run_name=run_name,
                hyperparameters={
                    "C": C_val,
                    "max_iter": max_iter,
                    "solver": solver,
                },
                metrics=metrics_test,
                train_time=train_time,
                test_time=test_time,
            )
            status.text("")

    # ══════════════════════════════════════════
    #  LTSM
    # ══════════════════════════════════════════

    elif model_type == "LSTM":

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import time

        from src.preprocessing.rnn_preprocessing import prepare_rnn_dataloaders
        from src.rnn_model import LSTMClassifier
        from src.engine import train_step, eval_step, evaluate_metrics
        from src.experiment_logger import save_experiment_log

        # ===========================
        # Hyperparameters
        # ===========================

        st.sidebar.header("Siêu tham số")

        batch_size = st.sidebar.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=2,
            key="lstm_batch"
        )

        epochs = st.sidebar.slider(
            "Epochs",
            5,
            100,
            30,
            step=5,
            key="lstm_epoch"
        )

        learning_rate = st.sidebar.selectbox(
            "Learning Rate",
            [0.01,0.005,0.001,0.0005,0.0001],
            index=2,
            key="lstm_lr"
        )

        st.sidebar.header("Kiến trúc LSTM")

        embedding_dim = st.sidebar.selectbox(
            "Embedding Dimension",
            [16,32,64],
            index=1
        )

        hidden_size = st.sidebar.selectbox(
            "Hidden Size",
            [32,64,128],
            index=1
        )

        num_layers = st.sidebar.selectbox(
            "Number of Layers",
            [1,2,3],
            index=1
        )

        dropout = st.sidebar.slider(
            "Dropout",
            0.0,
            0.5,
            0.3,
            step=0.05
        )

        fc_hidden_size = st.sidebar.selectbox(
            "FC Hidden Size",
            [32, 64, 128],
            index=1,
        )

        bidirectional = st.sidebar.checkbox(
            "Bidirectional",
            value=False
        )

        # ===========================
        # Architecture
        # ===========================

        st.subheader("Kiến trúc mô hình")

        arch_text = f"""
        | Layer | Detail |
        |---|---|
        | Embedding | vocab → {embedding_dim} |
        | Embedding Dropout | {dropout} |
        | LSTM | hidden={hidden_size}, layers={num_layers} |
        | Bidirectional | {bidirectional} |
        | FC1 | {hidden_size * (2 if bidirectional else 1)} → {fc_hidden_size} |
        | Activation | ReLU |
        | Classifier Dropout | {dropout} |
        | FC2 | {fc_hidden_size} → 1 |
        | Output | Sigmoid |
        | Loss | BCELoss |
        | Optimizer | Adam (lr={learning_rate}) |
        """

        st.markdown(arch_text)

        # ===========================
        # Train
        # ===========================

        if st.button("🚀 Bắt đầu Huấn luyện", use_container_width=True):

            progress = st.progress(0)
            status = st.empty()

            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("Loss")
                loss_chart = st.empty()

            with col_r:
                st.subheader("Accuracy")
                acc_chart = st.empty()

            status.text("Đang tokenize dữ liệu...")

            train_loader, val_loader, test_loader, metadata = \
                prepare_rnn_dataloaders(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    batch_size=batch_size
                )

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            model = LSTMClassifier(
                vocab_size=metadata["vocab_size"],
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                fc_hidden_size=fc_hidden_size,
            ).to(device)

            criterion = nn.BCELoss()

            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate
            )

            history = {
                "train_loss": [],
                "val_loss": [],

                "train_acc": [],
                "val_acc": [],

                "train_precision": [],
                "val_precision": [],

                "train_recall": [],
                "val_recall": [],

                "train_f1": [],
                "val_f1": [],
            }

            best_val_loss = float("inf")

            os.makedirs("models", exist_ok=True)

            status.text(f"Huấn luyện trên: {device}")

            train_start = time.perf_counter()

            for epoch in range(epochs):

                t_loss, t_acc, t_pre, t_rec, t_f1 = train_step(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )

                v_loss, v_acc, v_pre, v_rec, v_f1, _, _ = eval_step(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                history["train_loss"].append(t_loss)
                history["val_loss"].append(v_loss)

                history["train_acc"].append(t_acc)
                history["val_acc"].append(v_acc)

                history["train_precision"].append(t_pre)
                history["val_precision"].append(v_pre)

                history["train_recall"].append(t_rec)
                history["val_recall"].append(v_rec)

                history["train_f1"].append(t_f1)
                history["val_f1"].append(v_f1)

                if v_loss < best_val_loss:

                    best_val_loss = v_loss

                    torch.save(
                        model.state_dict(),
                        "models/temp_best_lstm_model.pth"
                    )

                loss_chart.line_chart(
                    pd.DataFrame({
                        "Train Loss": history["train_loss"],
                        "Val Loss": history["val_loss"]
                    })
                )

                acc_chart.line_chart(
                    pd.DataFrame({
                        "Train Acc": history["train_acc"],
                        "Val Acc": history["val_acc"]
                    })
                )

                progress.progress((epoch + 1) / epochs)

                status.text(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss {t_loss:.4f} | "
                    f"Val Loss {v_loss:.4f} | "
                    f"Val Acc {v_acc:.4f}"
                )
                time.sleep(0.02)

            train_end = time.perf_counter()
            st.success(f"✅ Hoàn tất! Best Val Loss: {best_val_loss:.4f}. Đang đánh giá trên tập Test...")

            model.load_state_dict(
                torch.load("models/temp_best_lstm_model.pth")
            )

            model.eval()

            test_start = time.perf_counter()

            test_loss, test_acc, _, _, _, preds, targets = eval_step(
                model,
                test_loader,
                criterion,
                device,
            )

            test_end = time.perf_counter()

            metrics = evaluate_metrics(targets, preds)
            metrics["Loss"] = test_loss

            # ===========================
            # Ghi log thực nghiệm
            # ===========================

            run_name = st.session_state.exp_name.strip()

            if not run_name:
                run_name = datetime.now().strftime("Run_%H_%M_%S")
                st.session_state.exp_name = run_name

            hp_str = (
                f"batch={batch_size}, "
                f"epochs={epochs}, "
                f"lr={learning_rate}, "
                f"embed={embedding_dim}, "
                f"hidden={hidden_size}, "
                f"layers={num_layers}, "
                f"dropout={dropout}, "
                f"fc_hidden={fc_hidden_size}, "
                f"bidirectional={bidirectional}"
            )

            is_new_best, global_best_acc = save_experiment_log(
                model_name="LSTM",
                run_name=run_name,
                history=history,
                test_metrics=metrics,
                hyperparameters=hp_str,
                train_time=train_end - train_start,
                test_time=test_end - test_start,
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                test_size=len(test_loader.dataset),
                device=str(device),
                model=model,
            )

            # ===========================
            # Lưu best model
            # ===========================

            st.session_state["lstm_config"] = {
                "vocab_size": metadata["vocab_size"],
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "fc_hidden_size": fc_hidden_size,
                "bidirectional": bidirectional,
            }

            if is_new_best:

                if os.path.exists("models/temp_best_lstm_model.pth"):
                    os.replace(
                        "models/temp_best_lstm_model.pth",
                        "models/best_lstm_model.pth"
                    )

                with open("models/best_lstm_config.json", "w") as f:
                    json.dump(
                        st.session_state["lstm_config"],
                        f,
                        indent=4
                    )

                st.success(f"🏆 Cấu hình này đã đạt Test Accuracy tốt nhất ({metrics['Accuracy']:.4f}) và được lưu làm Best Model!")

            else:

                if os.path.exists("models/temp_best_lstm_model.pth"):
                    os.remove("models/temp_best_lstm_model.pth")

                st.info(f"💾 Test Accuracy {metrics['Accuracy']:.4f} (Kỷ lục LTSM: {global_best_acc:.4f}). Kết quả đã lưu lịch sử.")
            
    # ══════════════════════════════════════════
    #  GRU
    # ══════════════════════════════════════════

    elif model_type == "GRU":

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import time

        from src.preprocessing.rnn_preprocessing import prepare_rnn_dataloaders
        from src.rnn_model import GRUClassifier
        from src.engine import train_step, eval_step, evaluate_metrics
        from src.experiment_logger import save_experiment_log

        # ===========================
        # Hyperparameters
        # ===========================

        st.sidebar.header("Siêu tham số")

        batch_size = st.sidebar.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=2,
            key="gru_batch"
        )

        epochs = st.sidebar.slider(
            "Epochs",
            5,
            100,
            30,
            step=5,
            key="gru_epoch"
        )

        learning_rate = st.sidebar.selectbox(
            "Learning Rate",
            [0.01,0.005,0.001,0.0005,0.0001],
            index=2,
            key="gru_lr"
        )

        st.sidebar.header("Kiến trúc GRU")

        embedding_dim = st.sidebar.selectbox(
            "Embedding Dimension",
            [16,32,64],
            index=1
        )

        hidden_size = st.sidebar.selectbox(
            "Hidden Size",
            [32,64,128],
            index=1
        )

        num_layers = st.sidebar.selectbox(
            "Number of Layers",
            [1,2,3],
            index=1
        )

        dropout = st.sidebar.slider(
            "Dropout",
            0.0,
            0.5,
            0.3,
            step=0.05
        )

        fc_hidden_size = st.sidebar.selectbox(
            "FC Hidden Size",
            [32, 64, 128],
            index=1,
        )

        bidirectional = st.sidebar.checkbox(
            "Bidirectional",
            value=False
        )

        # ===========================
        # Architecture
        # ===========================

        st.subheader("Kiến trúc mô hình")

        arch_text = f"""
        | Layer | Detail |
        |---|---|
        | Embedding | vocab → {embedding_dim} |
        | Embedding Dropout | {dropout} |
        | GRU | hidden={hidden_size}, layers={num_layers} |
        | Bidirectional | {bidirectional} |
        | FC1 | {hidden_size * (2 if bidirectional else 1)} → {fc_hidden_size} |
        | Activation | ReLU |
        | Classifier Dropout | {dropout} |
        | FC2 | {fc_hidden_size} → 1 |
        | Output | Sigmoid |
        | Loss | BCELoss |
        | Optimizer | Adam (lr={learning_rate}) |
        """

        st.markdown(arch_text)

        # ===========================
        # Train
        # ===========================

        if st.button("🚀 Bắt đầu Huấn luyện", use_container_width=True):

            progress = st.progress(0)
            status = st.empty()

            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("Loss")
                loss_chart = st.empty()

            with col_r:
                st.subheader("Accuracy")
                acc_chart = st.empty()

            status.text("Đang tokenize dữ liệu...")

            train_loader, val_loader, test_loader, metadata = \
                prepare_rnn_dataloaders(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    batch_size=batch_size
                )

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            model = GRUClassifier(
                vocab_size=metadata["vocab_size"],
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                fc_hidden_size=fc_hidden_size,
            ).to(device)

            criterion = nn.BCELoss()

            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate
            )

            history = {
                "train_loss": [],
                "val_loss": [],

                "train_acc": [],
                "val_acc": [],

                "train_precision": [],
                "val_precision": [],

                "train_recall": [],
                "val_recall": [],

                "train_f1": [],
                "val_f1": [],
            }

            best_val_loss = float("inf")

            os.makedirs("models", exist_ok=True)

            status.text(f"Huấn luyện trên: {device}")
            train_start = time.perf_counter()

            for epoch in range(epochs):

                t_loss, t_acc, t_pre, t_rec, t_f1 = train_step(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                )

                v_loss, v_acc, v_pre, v_rec, v_f1, _, _ = eval_step(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                history["train_loss"].append(t_loss)
                history["val_loss"].append(v_loss)

                history["train_acc"].append(t_acc)
                history["val_acc"].append(v_acc)

                history["train_precision"].append(t_pre)
                history["val_precision"].append(v_pre)

                history["train_recall"].append(t_rec)
                history["val_recall"].append(v_rec)

                history["train_f1"].append(t_f1)
                history["val_f1"].append(v_f1)

                if v_loss < best_val_loss:

                    best_val_loss = v_loss

                    torch.save(
                        model.state_dict(),
                        "models/temp_best_gru_model.pth"
                    )

                loss_chart.line_chart(
                    pd.DataFrame({
                        "Train Loss": history["train_loss"],
                        "Val Loss": history["val_loss"]
                    })
                )

                acc_chart.line_chart(
                    pd.DataFrame({
                        "Train Acc": history["train_acc"],
                        "Val Acc": history["val_acc"]
                    })
                )

                progress.progress((epoch + 1) / epochs)

                status.text(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss {t_loss:.4f} | "
                    f"Val Loss {v_loss:.4f} | "
                    f"Val Acc {v_acc:.4f}"
                )
                time.sleep(0.02)

            train_end = time.perf_counter()

            st.success(f"✅ Hoàn tất! Best Val Loss: {best_val_loss:.4f}. Đang đánh giá trên tập Test...")

            model.load_state_dict(
                torch.load("models/temp_best_gru_model.pth")
            )

            model.eval()

            test_start = time.perf_counter()

            test_loss, test_acc, _, _, _, preds, targets = eval_step(
                model,
                test_loader,
                criterion,
                device,
            )

            test_end = time.perf_counter()

            metrics = evaluate_metrics(targets, preds)
            metrics["Loss"] = test_loss

            # ===========================
            # Ghi log thực nghiệm
            # ===========================

            run_name = st.session_state.exp_name.strip()

            if not run_name:
                run_name = datetime.now().strftime("Run_%H_%M_%S")
                st.session_state.exp_name = run_name

            hp_str = (
                f"batch={batch_size}, "
                f"epochs={epochs}, "
                f"lr={learning_rate}, "
                f"embed={embedding_dim}, "
                f"hidden={hidden_size}, "
                f"layers={num_layers}, "
                f"dropout={dropout}, "
                f"fc_hidden={fc_hidden_size}, "
                f"bidirectional={bidirectional}"
            )

            is_new_best, global_best_acc = save_experiment_log(
                model_name="GRU",
                run_name=run_name,
                history=history,
                test_metrics=metrics,
                hyperparameters=hp_str,
                train_time=train_end - train_start,
                test_time=test_end - test_start,
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                test_size=len(test_loader.dataset),
                device=str(device),
                model=model,
            )

            # ===========================
            # Lưu best model
            # ===========================

            st.session_state["gru_config"] = {
                "vocab_size": metadata["vocab_size"],
                "embedding_dim": embedding_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "fc_hidden_size": fc_hidden_size,
                "bidirectional": bidirectional,
            }

            if is_new_best:

                if os.path.exists("models/temp_best_gru_model.pth"):
                    os.replace(
                        "models/temp_best_gru_model.pth",
                        "models/best_gru_model.pth"
                    )

                with open("models/best_gru_config.json", "w") as f:
                    json.dump(
                        st.session_state["gru_config"],
                        f,
                        indent=4
                    )

                st.success(f"🏆 Cấu hình này đã đạt Test Accuracy tốt nhất ({metrics['Accuracy']:.4f}) và được lưu làm Best Model!")

            else:

                if os.path.exists("models/temp_best_gru_model.pth"):
                    os.remove("models/temp_best_gru_model.pth")

                st.info(f"💾 Test Accuracy {metrics['Accuracy']:.4f} (Kỷ lục GRU: {global_best_acc:.4f}). Kết quả đã lưu lịch sử.")

# ═══════════════════════════════════════════════
#  TRANG 3: ĐÁNH GIÁ & SO SÁNH
# ═══════════════════════════════════════════════

elif page == "📊 Đánh giá":

    import seaborn as sns
    import ast

    st.title("📊 Đánh giá & So sánh mô hình")

    log_file = "experiments/experiments_log.csv"

    if not os.path.exists(log_file):
        st.warning("Chưa có dữ liệu thực nghiệm.")
        st.stop()

    df_logs = pd.read_csv(log_file)

    if df_logs.empty:
        st.warning("Chưa có dữ liệu thực nghiệm.")
        st.stop()

    # ==========================================================
    # Dashboard KPI
    # ==========================================================

    st.header("1. Tổng quan thực nghiệm")

    best_idx = df_logs["Test Accuracy"].idxmax()
    best_row = df_logs.loc[best_idx]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Số Experiments",
            len(df_logs)
        )

    with c2:
        st.metric(
            "Best Accuracy",
            f"{best_row['Test Accuracy']:.4f}"
        )

    with c3:
        st.metric(
            "Training Time TB",
            f"{df_logs['Training Time (s)'].mean():.2f}s"
        )

    with c4:
        st.metric(
            "Parameters TB",
            f"{int(df_logs['Parameters'].mean()):,}"
        )

    st.info(
        f"🏆 Best Run: **{best_row['Run Name']}** "
        f"({best_row['Model']})"
    )

    st.divider()

    # ==========================================================
    # Bảng thực nghiệm
    # ==========================================================

    st.header("2. Bảng so sánh")

    sort_col = st.selectbox(
        "Sắp xếp theo",
        [
            "Test Accuracy",
            "Best Validation Accuracy",
            "Training Time (s)",
            "Parameters",
            "Accuracy Gap"
        ]
    )

    ascending = st.checkbox(
        "Tăng dần",
        value=False
    )

    df_sorted = (
        df_logs
        .sort_values(sort_col, ascending=ascending)
        .reset_index(drop=True)
    )

    display_cols = [
        "Model",
        "Run Name",
        "Device",

        "Training Time (s)",
        "Testing Time (s)",

        "Parameters",

        "Best Validation Accuracy",
        "Test Accuracy",

        "Accuracy Gap"
    ]

    display_cols = [
        c for c in display_cols
        if c in df_sorted.columns
    ]

    st.dataframe(
        df_sorted[display_cols],
        use_container_width=True
    )

    st.divider()

    # ==========================================================
    # Chuẩn bị Label
    # ==========================================================

    df_sorted["Label"] = (
        df_sorted["Model"] +
        " - " +
        df_sorted["Run Name"]
    )

    labels = df_sorted["Label"].tolist()

    # ==========================================================
    # Test Metrics
    # ==========================================================

    st.header("3. So sánh Test Metrics")

    metric_cols = [
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1"
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(labels) * 2), 5)
    )

    x = np.arange(len(labels))

    width = 0.18

    colors = BAR_COLORS

    for i, col in enumerate(metric_cols):

        vals = df_sorted[col].astype(float).tolist()

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=col,
            color=colors[i]
        )

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height()+0.005,
                f"{v:.3f}",
                ha="center",
                fontsize=7
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.set_ylim(0,1.08)

    ax.legend()

    ax.set_ylabel("Score")

    st.pyplot(fig)

    st.divider()

    # ==========================================================
    # Time Comparison
    # ==========================================================

    st.header("4. Thời gian huấn luyện")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        labels,
        df_sorted["Training Time (s)"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        color=COLOR_BAR
    )

    # Hiển thị giá trị tại từng điểm
    for x, y in zip(labels, df_sorted["Training Time (s)"]):
        ax.text(
            x,
            y + 0.02 * df_sorted["Training Time (s)"].max(),
            f"{y:.2f}",
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL
        )

    ax.set_ylabel("Seconds")
    ax.set_xlabel("Experiments")

    # Giao diện trung tính
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=35, ha="right")

    st.pyplot(fig)

    st.divider()

    # ==========================================================
    # Parameters
    # ==========================================================

    st.header("5. Số lượng tham số")

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.bar(
        labels,
        df_sorted["Parameters"],
        color=COLOR_NEUTRAL
    )

    ax.set_ylabel("Parameters")

    plt.xticks(rotation=35, ha="right")

    st.pyplot(fig)

    st.divider()

    # ==========================================================
    # Gap Analysis
    # ==========================================================

    st.header("6. Overfitting Analysis")

    gap_cols = [
        "Accuracy Gap",
        "Precision Gap",
        "Recall Gap",
        "F1 Gap"
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(labels)*2),5)
    )

    x = np.arange(len(labels))

    width = 0.18

    for i,col in enumerate(gap_cols):

        vals = df_sorted[col].astype(float)

        ax.bar(
            x+i*width,
            vals,
            width,
            label=col,
            color = BAR_COLORS[i]
        )

    ax.axhline(
        0,
        color="black"
    )

    ax.set_xticks(x+width*1.5)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.legend()

    st.pyplot(fig)

    st.divider()

    # ==========================================================
    # Chi tiết từng cấu hình
    # ==========================================================

    st.header("7. Chi tiết từng cấu hình")

    selected = st.selectbox(
        "Chọn Experiment",
        df_sorted["Label"]
    )

    row = df_sorted[
        df_sorted["Label"] == selected
    ].iloc[0]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 General",
        "📈 Best Epoch",
        "📉 Worst Epoch",
        "📋 Average"
    ])

    # ==========================================================
    # GENERAL
    # ==========================================================

    with tab1:

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Thông tin")

            general = pd.DataFrame({
                "Thuộc tính":[
                    "Model",
                    "Run Name",
                    "Device",
                    "Training Time (s)",
                    "Testing Time (s)",
                    "Parameters",
                    "Train Samples",
                    "Validation Samples",
                    "Test Samples"
                ],
                "Giá trị":[
                    row["Model"],
                    row["Run Name"],
                    row["Device"],
                    f"{row['Training Time (s)']:.2f}",
                    f"{row['Testing Time (s)']:.3f}",
                    f"{int(row['Parameters']):,}",
                    int(row["Train Samples"]),
                    int(row["Validation Samples"]),
                    int(row["Test Samples"])
                ]
            })

            st.table(general)

            st.subheader("Hyperparameters")

            hp = str(row["Hyperparameters"])

            if hp != "nan":

                hp_pairs = [
                    x.strip()
                    for x in hp.split(",")
                ]

                hp_table = pd.DataFrame({
                    "Parameter":[
                        x.split("=")[0]
                        for x in hp_pairs
                        if "=" in x
                    ],
                    "Value":[
                        x.split("=")[1]
                        for x in hp_pairs
                        if "=" in x
                    ]
                })

                st.table(hp_table)

        with c2:

            st.subheader("Confusion Matrix")

            try:

                cm = np.array(
                    ast.literal_eval(
                        row["Confusion Matrix"]
                    )
                )

                fig, ax = plt.subplots(figsize=(5,4))

                sns.heatmap(
                    cm,
                    annot=True,
                    cmap="Blues",
                    fmt="d",
                    xticklabels=[
                        "Edible",
                        "Poisonous"
                    ],
                    yticklabels=[
                        "Edible",
                        "Poisonous"
                    ],
                    ax=ax
                )

                st.pyplot(fig)

            except Exception as e:

                st.error(e)

            if "Classification Report" in row:

                st.subheader("Classification Report")

                st.text(row["Classification Report"])

    # ==========================================================
    # BEST EPOCH
    # ==========================================================

    with tab2:

        best_table = pd.DataFrame({

            "Metric":[

                "Train Loss",
                "Validation Loss",

                "Train Accuracy",
                "Validation Accuracy",

                "Train Precision",
                "Validation Precision",

                "Train Recall",
                "Validation Recall",

                "Train F1",
                "Validation F1"

            ],

            "Value":[

                row["Best Train Loss"],
                row["Best Validation Loss"],

                row["Best Train Accuracy"],
                row["Best Validation Accuracy"],

                row["Best Train Precision"],
                row["Best Validation Precision"],

                row["Best Train Recall"],
                row["Best Validation Recall"],

                row["Best Train F1"],
                row["Best Validation F1"]

            ]

        })

        st.table(best_table)

    # ==========================================================
    # WORST EPOCH
    # ==========================================================

    with tab3:

        worst_table = pd.DataFrame({

            "Metric":[

                "Train Loss",
                "Validation Loss",

                "Train Accuracy",
                "Validation Accuracy",

                "Train Precision",
                "Validation Precision",

                "Train Recall",
                "Validation Recall",

                "Train F1",
                "Validation F1"

            ],

            "Value":[

                row["Worst Train Loss"],
                row["Worst Validation Loss"],

                row["Worst Train Accuracy"],
                row["Worst Validation Accuracy"],

                row["Worst Train Precision"],
                row["Worst Validation Precision"],

                row["Worst Train Recall"],
                row["Worst Validation Recall"],

                row["Worst Train F1"],
                row["Worst Validation F1"]

            ]

        })

        st.table(worst_table)

    # ==========================================================
    # AVERAGE
    # ==========================================================

    with tab4:

        avg_table = pd.DataFrame({

            "Metric":[

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

                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Test F1",

                "Accuracy Gap",
                "Precision Gap",
                "Recall Gap",
                "F1 Gap"

            ],

            "Value":[

                row["Average Train Loss"],
                row["Average Validation Loss"],

                row["Average Train Accuracy"],
                row["Average Validation Accuracy"],

                row["Average Train Precision"],
                row["Average Validation Precision"],

                row["Average Train Recall"],
                row["Average Validation Recall"],

                row["Average Train F1"],
                row["Average Validation F1"],

                row["Test Accuracy"],
                row["Test Precision"],
                row["Test Recall"],
                row["Test F1"],

                row["Accuracy Gap"],
                row["Precision Gap"],
                row["Recall Gap"],
                row["F1 Gap"]

            ]

        })

        st.table(avg_table)


