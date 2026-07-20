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


def render_validation_checks(validation, meta, y_train, y_val, y_test):
    all_labels = set(y_train.unique()) | set(y_val.unique()) | set(y_test.unique())
    checks = [
        ("Không còn ký hiệu '?' chưa xử lý", meta['missing_after'] == 0),
        ("Không còn cột veil-type", 'veil-type' in meta.get('dropped_columns', [])),
        ("Nhãn chỉ gồm 'e' và 'p'", all_labels == {'e', 'p'}),
        ("Tổng mẫu sau chia = tổng ban đầu", len(y_train) + len(y_val) + len(y_test) == meta['samples_before']),
        ("Dữ liệu thô hợp lệ", validation['status'] == 'pass'),
    ]
    for label, passed in checks:
        icon = "✅" if passed else "❌"
        css = "validation-pass" if passed else "validation-fail"
        st.markdown(f'<div class="{css}">{icon} {label}</div>', unsafe_allow_html=True)


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

    # ── 6. So sánh trước / sau ──
    st.header("6 · So sánh trước & sau tiền xử lý")
    comp = pd.DataFrame({
        "Chỉ tiêu": ["Số mẫu", "Số đặc trưng", "Ký hiệu '?'", "Cột hằng số", "Số lớp"],
        "Trước": [metadata["samples_before"], metadata["features_before"], metadata["missing_before"], len(metadata["constant_columns"]), 2],
        "Sau": [metadata["samples_after"], metadata["features_after"], metadata["missing_after"], 0, 2]
    })
    st.table(comp)
    st.markdown("---")

    # ── 7. Train / Val / Test ──
    st.header("7 · Phân chia tập dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train (70%)", f"{metadata['split_counts']['train']:,}")
    c2.metric("Validation (15%)", f"{metadata['split_counts']['validation']:,}")
    c3.metric("Test (15%)", f"{metadata['split_counts']['test']:,}")
    st.pyplot(plot_split_distribution(y_train, y_val, y_test))
    st.markdown("---")

    # ── 8. Validation checks ──
    st.header("8 · Kiểm tra toàn vẹn dữ liệu")
    render_validation_checks(validation_results, metadata, y_train, y_val, y_test)
    st.markdown("---")

    # ── 9. Metadata ──
    st.header("9 · Metadata")
    with st.expander("Xem preprocessing_metadata.json"):
        st.json(metadata)

# ═══════════════════════════════════════════════
#  TRANG 2: HUẤN LUYỆN CNN
# ═══════════════════════════════════════════════

elif page == "⚙️ Huấn luyện":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time

    from src.preprocessing.cnn_preprocessing import prepare_cnn_dataloaders
    from src.cnn_model import MushroomCNN1D
    from src.engine import train_step, eval_step

    st.title("⚙️ Huấn luyện CNN 1D")

    # ── Sidebar: Hyperparameters ──
    st.sidebar.header("Tên thực nghiệm")
    exp_name = st.sidebar.text_input("Experiment Name", value=datetime.now().strftime("Run_%H_%M_%S"))
    st.session_state['exp_name'] = exp_name

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
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        os.makedirs("models", exist_ok=True)

        # Lưu cấu hình vào session_state
        st.session_state['cnn_config'] = {
            'vocab_size': len(vocab), 'embed_dim': embed_dim,
            'num_filters': num_filters, 'kernel_size': kernel_size,
            'dropout_rate': dropout_rate, 'use_batch_norm': use_batch_norm
        }
        st.session_state['vocab'] = vocab

        status.text(f"Huấn luyện trên: {device}")

        for epoch in range(epochs):
            t_loss, t_acc = train_step(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, _, _ = eval_step(model, val_loader, criterion, device)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)

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

        st.success(f"✅ Hoàn tất! Best Val Loss: {best_val_loss:.4f}. Đang đánh giá trên tập Test...")

        # Đánh giá trên tập test để lưu log
        from src.engine import evaluate_metrics
        model.load_state_dict(torch.load("models/temp_best_cnn_model.pth"))
        model.eval()
        
        _, test_acc, preds, targets = eval_step(model, test_loader, criterion, device)
        metrics = evaluate_metrics(targets, preds)
        
        # Ghi log
        os.makedirs("experiments", exist_ok=True)
        log_file = "experiments/experiments_log.csv"
        file_exists = os.path.isfile(log_file)
        
        global_best_acc = 0.0
        if file_exists:
            try:
                df_logs = pd.read_csv(log_file)
                if not df_logs.empty and 'Test Acc' in df_logs.columns:
                    global_best_acc = df_logs['Test Acc'].max()
            except Exception:
                pass
                
        is_new_best = metrics['Accuracy'] > global_best_acc
        run_name = st.session_state.get('exp_name', "Run")
        cm_str = str(metrics['Confusion Matrix'].tolist())
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Run Name", "Batch Size", "Epochs", "LR", "Embed Dim", "Filters", "Kernel", "Dropout", "BatchNorm", "Best Val Loss", "Test Acc", "Precision", "Recall", "F1", "Confusion Matrix"])
            writer.writerow([run_name, batch_size, epochs, learning_rate, embed_dim, num_filters, kernel_size, dropout_rate, use_batch_norm, f"{best_val_loss:.4f}", f"{metrics['Accuracy']:.4f}", f"{metrics['Precision']:.4f}", f"{metrics['Recall']:.4f}", f"{metrics['F1-Score']:.4f}", cm_str])
        
        if is_new_best:
            if os.path.exists("models/temp_best_cnn_model.pth"):
                os.rename("models/temp_best_cnn_model.pth", "models/best_cnn_model.pth")
            with open("models/best_cnn_config.json", "w") as f:
                json.dump(st.session_state['cnn_config'], f)
            st.success(f"🏆 Cấu hình này đã đạt Test Accuracy tốt nhất ({metrics['Accuracy']:.4f}) và được lưu làm Best Model!")
        else:
            if os.path.exists("models/temp_best_cnn_model.pth"):
                os.remove("models/temp_best_cnn_model.pth")
            st.info(f"💾 Cấu hình này đạt Test Accuracy {metrics['Accuracy']:.4f} (Kỷ lục: {global_best_acc:.4f}). Kết quả đã được lưu lịch sử nhưng không ghi đè Best Model.")

        st.info("Chuyển sang trang **📊 Đánh giá** để xem lịch sử và so sánh kết quả.")

# ═══════════════════════════════════════════════
#  TRANG 3: ĐÁNH GIÁ & SO SÁNH
# ═══════════════════════════════════════════════

elif page == "📊 Đánh giá":
    import seaborn as sns

    st.title("📊 Đánh giá & So sánh Mô hình")

    log_file = "experiments/experiments_log.csv"
    if not os.path.exists(log_file):
        st.warning("Chưa có dữ liệu thực nghiệm. Vui lòng huấn luyện ít nhất một cấu hình.")
    else:
        df_logs = pd.read_csv(log_file)
        if df_logs.empty:
            st.warning("Chưa có dữ liệu thực nghiệm.")
        else:
            df_sorted = df_logs.sort_values(by="Test Acc", ascending=False).reset_index(drop=True)

            # ── 1. Bảng so sánh ──
            st.header("1 · Bảng so sánh thực nghiệm")
            display_cols = ["Run Name", "Batch Size", "Epochs", "LR", "Embed Dim",
                            "Filters", "Kernel", "Dropout", "BatchNorm",
                            "Best Val Loss", "Test Acc", "Precision", "Recall", "F1"]
            st.dataframe(df_sorted[display_cols], use_container_width=True)

            st.markdown("---")

            # ── 2. Biểu đồ so sánh Metrics ──
            st.header("2 · Biểu đồ so sánh Metrics")
            metric_cols = ["Test Acc", "Precision", "Recall", "F1"]
            run_names = df_sorted["Run Name"].tolist()

            fig, ax = plt.subplots(figsize=(max(8, len(run_names) * 2.5), 5))
            x = np.arange(len(run_names))
            width = 0.18
            colors = [COLOR_EDIBLE, COLOR_POISONOUS, COLOR_NEUTRAL, COLOR_BAR]

            for i, col in enumerate(metric_cols):
                vals = df_sorted[col].astype(float).tolist()
                bars = ax.bar(x + i * width, vals, width, label=col,
                              color=colors[i], edgecolor='white')
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha='center', va='bottom', fontsize=7.5)

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(run_names, rotation=45, ha='right')
            ax.set_ylabel("Giá trị")
            ax.set_title("So sánh Metrics giữa các cấu hình", fontweight='bold')
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.08)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown("---")

            # ── 3. Chi tiết từng cấu hình ──
            st.header("3 · Chi tiết từng cấu hình")
            selected_run = st.selectbox("Chọn cấu hình:", run_names)
            row = df_sorted[df_sorted["Run Name"] == selected_run].iloc[0]

            col_metric, col_cm = st.columns([1, 1])

            with col_metric:
                st.subheader("Metrics")
                metrics_table = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Best Val Loss"],
                    "Giá trị": [
                        f"{float(row['Test Acc']):.4f}",
                        f"{float(row['Precision']):.4f}",
                        f"{float(row['Recall']):.4f}",
                        f"{float(row['F1']):.4f}",
                        f"{float(row['Best Val Loss']):.4f}"
                    ]
                })
                st.table(metrics_table)

                st.subheader("Siêu tham số")
                hp_table = pd.DataFrame({
                    "Tham số": ["Batch Size", "Epochs", "Learning Rate",
                                "Embed Dim", "Filters", "Kernel", "Dropout", "BatchNorm"],
                    "Giá trị": [
                        str(row["Batch Size"]), str(row["Epochs"]), str(row["LR"]),
                        str(row["Embed Dim"]), str(row["Filters"]), str(row["Kernel"]),
                        str(row["Dropout"]), str(row["BatchNorm"])
                    ]
                })
                st.table(hp_table)

            with col_cm:
                st.subheader("Confusion Matrix")
                try:
                    cm = np.array(ast.literal_eval(str(row["Confusion Matrix"])))
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Edible (0)', 'Poisonous (1)'],
                                yticklabels=['Edible (0)', 'Poisonous (1)'], ax=ax_cm)
                    ax_cm.set_ylabel("Nhãn thực tế")
                    ax_cm.set_xlabel("Nhãn dự đoán")
                    ax_cm.set_title(f"Confusion Matrix – {selected_run}", fontweight='bold')
                    fig_cm.tight_layout()
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.error(f"Không thể hiển thị Confusion Matrix: {e}")




