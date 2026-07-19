# Đồ án Cuối kỳ - Nhập môn Học máy
**Chủ đề**: Phân loại nấm (Mushroom Classification) sử dụng CNN 1D.

## Cấu trúc thư mục

- `data/`: Chứa bộ dữ liệu `agaricus-lepiota.data`.
- `src/`: Mã nguồn chính của dự án.
  - `data_processing.py`: Load dữ liệu, mã hóa One-Hot, chia Train/Val/Test.
  - `cnn_model.py`: Định nghĩa kiến trúc mô hình CNN 1D bằng PyTorch.
  - `engine.py`: Vòng lặp huấn luyện và đánh giá mô hình.
  - `utils.py`: Các hàm tiện ích (vẽ Confusion Matrix).
- `app.py`: Giao diện ứng dụng Streamlit.
- `requirements.txt`: Các thư viện cần thiết.

## Cài đặt

1. Đảm bảo bạn đã cài đặt Python.
2. Cài đặt các thư viện yêu cầu:
   ```bash
   pip install -r requirements.txt
   ```

## Khởi chạy ứng dụng

Sử dụng lệnh sau để chạy giao diện Streamlit:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt. Tại đây, bạn có thể điều chỉnh siêu tham số và huấn luyện mô hình trực quan.

## Đặc điểm nhóm dự án
Cấu trúc code được thiết kế theo hướng module để hỗ trợ làm việc nhóm hiệu quả, tránh bị conflict khi merge code. Mỗi thành viên có thể phụ trách một module riêng biệt hoặc dễ dàng mở rộng thêm các mô hình khác (như RNN/LSTM) trong tương lai.
