import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import get_dataloaders
from src.cnn_model import MushroomCNN1D
from src.engine import train_step, eval_step, evaluate_metrics
from src.utils import plot_confusion_matrix

# Thiết lập cấu hình trang
st.set_page_config(page_title="Đồ Án Nhập Môn Học Máy", layout="wide", page_icon="🍄")

# ----------------- HÀM TIỆN ÍCH -----------------
@st.cache_data
def load_raw_data():
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", 
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", 
        "stalk-surface-below-ring", "stalk-color-above-ring", 
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", 
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    try:
        df = pd.read_csv("data/agaricus-lepiota.data", header=None, names=columns)
        return df
    except:
        return None

# ----------------- SIDEBAR MENU -----------------
st.sidebar.title("🍄 Menu Điều Hướng")
page = st.sidebar.radio(
    "Vui lòng chọn trang:",
    ["📄 Dashboard Báo Cáo", "⚙️ Huấn luyện & Thực nghiệm", "📊 Đánh giá & So sánh"]
)

# ----------------- TRANG 1: DASHBOARD BÁO CÁO -----------------
if page == "📄 Dashboard Báo Cáo":
    st.title("🍄 Đồ án: Phân loại Nấm (Mushroom Classification)")
    st.markdown("### 1. Giới thiệu bài toán và Mục tiêu")
    st.write("""
    Mục tiêu của đồ án là xây dựng một mô hình Học máy có khả năng phân loại nấm thành hai loại: **Ăn được (Edible)** hoặc **Có độc (Poisonous)** dựa trên các đặc điểm hình thái của chúng.
    Mô hình được áp dụng ở đây là **Mạng nơ-ron tích chập 1 chiều (CNN 1D)**.
    """)
    
    st.markdown("### 2. Mô tả bộ dữ liệu")
    df = load_raw_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số mẫu (Samples)", len(df))
        col2.metric("Số lượng đặc trưng (Features)", len(df.columns) - 1)
        
        # Đếm số lượng label
        edible_count = len(df[df['class'] == 'e'])
        poisonous_count = len(df[df['class'] == 'p'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='class', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xticklabels(['Poisonous (Độc)', 'Edible (Ăn được)'])
        ax.set_title("Phân bố nhãn dữ liệu")
        col3.pyplot(fig)
        
        st.write("**Một số dòng dữ liệu mẫu:**")
        st.dataframe(df.head(5))
    else:
        st.error("Không tìm thấy file dữ liệu tại `data/agaricus-lepiota.data`")

    st.markdown("### 3. Tiền xử lý dữ liệu")
    st.write("""
    - Toàn bộ dữ liệu dạng phân loại (categorical) được mã hóa bằng kỹ thuật **One-Hot Encoding**.
    - Tập dữ liệu được chia theo tỉ lệ: **70% Train, 15% Validation, 15% Test**.
    - Việc chia tập dữ liệu sử dụng chiến lược `stratify` để đảm bảo tỷ lệ Nấm Độc/Ăn được cân bằng ở mọi tập.
    """)
    
    st.markdown("### 4. Phương pháp và Kiến trúc mô hình CNN 1D")
    st.write("""
    - Do dữ liệu là dạng bảng (Tabular), chuỗi đặc trưng sau khi One-Hot Encoding được xem như một véc-tơ không gian 1 chiều.
    - Dữ liệu đầu vào được reshape thành kích thước `(batch_size, 1, sequence_length)` để đưa qua lớp `Conv1d`.
    """)
    st.code("""
Kiến trúc tổng quát của mô hình:
1. Conv1D (Trích xuất đặc trưng cục bộ)
2. Batch Normalization (Tùy chọn) + ReLU
3. MaxPool1D (Giảm chiều dữ liệu)
4. Flatten (Dàn phẳng)
5. Dropout (Chống Overfitting)
6. Fully Connected Layer 1 (32 units) + ReLU
7. Dropout
8. Fully Connected Layer 2 (1 unit) + Sigmoid (Phân loại nhị phân)
    """, language='text')

# ----------------- TRANG 2: HUẤN LUYỆN -----------------
elif page == "⚙️ Huấn luyện & Thực nghiệm":
    st.title("⚙️ Thiết lập Thực nghiệm & Huấn luyện")
    st.write("Tại đây, bạn có thể thay đổi các Siêu tham số (Hyperparameters) để quan sát sự ảnh hưởng đến tốc độ học và kết quả của mô hình.")
    
    st.sidebar.header("Cấu hình Siêu tham số")
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    epochs = st.sidebar.slider("Số lượng Epochs", min_value=5, max_value=100, value=20, step=5)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)

    st.sidebar.subheader("Kiến trúc CNN")
    num_filters = st.sidebar.slider("Số lượng Filters", 8, 64, 16, step=8)
    kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, step=2)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.1)
    use_batch_norm = st.sidebar.checkbox("Sử dụng Batch Normalization", value=False)
    
    if st.button("Bắt đầu Huấn luyện 🚀", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loss over Epochs")
            loss_chart = st.empty()
        with col2:
            st.subheader("Accuracy over Epochs")
            acc_chart = st.empty()
            
        status_text.text("Đang tải và chia tập dữ liệu...")
        try:
            train_loader, val_loader, test_loader, input_dim = get_dataloaders(
                data_path="data/agaricus-lepiota.data", batch_size=batch_size
            )
            
            # Lưu lại thông tin input_dim để dùng ở trang đánh giá
            st.session_state['input_dim'] = input_dim
        except Exception as e:
            st.error(f"Lỗi tải dữ liệu: {e}")
            st.stop()
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MushroomCNN1D(
            input_dim=input_dim, num_filters=num_filters, kernel_size=kernel_size, 
            dropout_rate=dropout_rate, use_batch_norm=use_batch_norm
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        os.makedirs("models", exist_ok=True)
        best_model_path = "models/best_cnn_model.pth"
        
        # Lưu các hyperparam của model hiện tại
        st.session_state['model_config'] = {
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm
        }
        
        status_text.text(f"Bắt đầu huấn luyện trên thiết bị: {device}")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = eval_step(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                
            df_loss = pd.DataFrame({'Train Loss': history['train_loss'], 'Val Loss': history['val_loss']})
            df_acc = pd.DataFrame({'Train Acc': history['train_acc'], 'Val Acc': history['val_acc']})
            
            loss_chart.line_chart(df_loss)
            acc_chart.line_chart(df_acc)
            
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            time.sleep(0.05)
            
        st.success(f"Huấn luyện hoàn tất! Mô hình tốt nhất đã được lưu tại `{best_model_path}`.")
        st.info("Hãy chuyển sang trang **Đánh giá & So sánh** để xem hiệu suất trên tập Test ẩn.")

# ----------------- TRANG 3: ĐÁNH GIÁ -----------------
elif page == "📊 Đánh giá & So sánh":
    st.title("📊 Đánh giá Mô hình trên Tập Test")
    
    best_model_path = "models/best_cnn_model.pth"
    if not os.path.exists(best_model_path) or 'input_dim' not in st.session_state:
        st.warning("Chưa tìm thấy mô hình đã huấn luyện. Vui lòng quay lại trang **Huấn luyện & Thực nghiệm** để chạy huấn luyện trước.")
    else:
        st.write("Đang tính toán các độ đo trên tập Test (dữ liệu mô hình chưa từng nhìn thấy trong quá trình học)...")
        
        # Load dữ liệu (cần test_loader)
        batch_size = 64
        _, _, test_loader, input_dim = get_dataloaders(data_path="data/agaricus-lepiota.data", batch_size=batch_size)
        
        # Load mô hình
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = st.session_state['model_config']
        
        model = MushroomCNN1D(
            input_dim=input_dim, 
            num_filters=config['num_filters'], 
            kernel_size=config['kernel_size'], 
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm']
        ).to(device)
        
        model.load_state_dict(torch.load(best_model_path))
        criterion = nn.BCELoss()
        
        test_loss, test_acc, test_preds, test_targets = eval_step(model, test_loader, criterion, device)
        metrics = evaluate_metrics(test_targets, test_preds)
        
        st.markdown("### Kết quả Độ đo (Metrics)")
        col3, col4, col5, col6 = st.columns(4)
        col3.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col4.metric("Precision", f"{metrics['Precision']:.4f}")
        col5.metric("Recall", f"{metrics['Recall']:.4f}")
        col6.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        st.markdown("### Confusion Matrix")
        cm_fig = plot_confusion_matrix(metrics['Confusion Matrix'])
        st.pyplot(cm_fig)
        
        st.markdown("### Phân tích Overfitting/Underfitting")
        st.write("""
        *Hãy kết hợp biểu đồ Loss/Accuracy ở trang Huấn luyện và bảng kết quả Test ở đây để phân tích. Nếu Train Loss rất thấp nhưng Test Loss cao và Accuracy trên Test thấp hơn Train nhiều, mô hình đang bị Overfitting.*
        """)
