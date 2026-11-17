# Dự đoán bệnh tim - Machine Learning + Streamlit

Dự án mẫu cho đồ án: phân tích dữ liệu bệnh tim, trực quan hóa, so sánh thuật toán phân lớp, đánh giá mô hình, lưu pipeline đã huấn luyện và chạy ứng dụng minh họa.

> **Cảnh báo:** Chỉ phục vụ học tập/nghiên cứu. Không phải thiết bị hay công cụ chẩn đoán y khoa, không dùng để tự chẩn đoán hoặc quyết định điều trị.

## 1. Dữ liệu

Dự án dùng tệp `data/raw/processed.cleveland.data` thuộc **UCI Heart Disease dataset**, Cleveland subset. Bộ này có 303 dòng, 13 thuộc tính đầu vào và nhãn gốc `num` có các mức 0-4. Theo thiết lập phổ biến của bộ dữ liệu, chuyển về nhị phân:

- `0`: `num == 0` - không ghi nhận bệnh theo nhãn dữ liệu.
- `1`: `num > 0` - có ghi nhận bệnh theo nhãn dữ liệu (gộp các mức gốc 1-4).

Hai trường có giá trị thiếu: `ca` có 4 giá trị, `thal` có 2 giá trị. Các giá trị `?` được đọc thành missing và xử lý bên trong pipeline để tránh xóa mẫu: median cho biến liên tục; giá trị xuất hiện nhiều nhất cho biến phân loại.

Nguồn chính thức: https://archive.ics.uci.edu/dataset/45/heart+disease

## 2. Thuật toán được so sánh

`src/heart_disease_classification/train_model.py` so sánh 8 ứng viên bằng cùng một quy trình tiền xử lý và kiểm định chéo phân tầng 5-fold:

1. **Logistic Regression** - mô hình tuyến tính cho phân lớp; dễ giải thích hơn nhiều mô hình phi tuyến.
2. **Decision Tree** - tạo các luật phân nhánh; dễ minh họa nhưng có thể overfit.
3. **Random Forest** - tập hợp nhiều cây quyết định với cơ chế bagging.
4. **Extra Trees** - tập hợp cây ngẫu nhiên, dùng làm mô hình đối chứng với Random Forest.
5. **SVM (RBF)** - tìm ranh giới phân lớp dựa trên support vectors và kernel.
6. **K-Nearest Neighbors** - dự đoán dựa trên nhãn của các điểm gần nhất; cần chuẩn hóa thang đo.
7. **Gaussian Naive Bayes** - mô hình xác suất đơn giản dựa trên giả định độc lập có điều kiện.
8. **Gradient Boosting** - xây dựng tuần tự nhiều cây yếu để cải thiện dự đoán.

Tài liệu thuật toán: scikit-learn User Guide (các liên kết trong `reports/report.md`).

## 3. Quy trình thực nghiệm

- Giữ nguyên 303 dòng; không xóa các mẫu bị thiếu.
- Tách train/test theo tỷ lệ 80/20, `stratify=y`, `random_state=42`.
- Numeric: median imputation + StandardScaler. Categorical: most-frequent imputation + One-Hot Encoding.
- Đánh giá các ứng viên bằng Stratified 5-fold CV trên tập train.
- Chọn theo **F1 trung bình CV cao nhất**; test set không được dùng làm tiêu chí chọn mô hình.
- Báo cáo Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix và ROC curve trên test.
- Sau khi đánh giá holdout, fit lại pipeline đã chọn trên toàn bộ 303 mẫu để tạo artifact dùng trong app. Vì vậy, metric trong báo cáo là từ mô hình được đánh giá trên holdout trước khi fit lại, không phải metric đo trên chính tập đã dùng để fit artifact triển khai.

## 4. Kết quả của lần chạy đã đóng gói

Mô hình được chọn: **Logistic Regression** (theo F1 CV).

| Chỉ số | 5-fold CV trung bình trên train | Holdout test |
|---|---:|---:|
| Accuracy | 0.855 | 0.869 |
| Precision | 0.873 | 0.813 |
| Recall | 0.810 | 0.929 |
| F1 | 0.837 | 0.867 |
| ROC-AUC | 0.902 | 0.966 |

Confusion matrix trên tập test, theo thứ tự `[[TN, FP], [FN, TP]]`: `[[27, 6], [2, 26]]`.

Tập test chỉ có 61 mẫu, do đó các chỉ số là ước lượng không ổn định và có thể thay đổi với cách chia khác. Không xem ROC-AUC cao là bằng chứng mô hình đã hiệu chuẩn tốt hoặc đủ năng lực lâm sàng.

## 5. Cài đặt và chạy trên Windows

Mở Terminal/PowerShell tại thư mục gốc của dự án.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Chạy lại pipeline huấn luyện, sinh mô hình và biểu đồ:

```powershell
python -m src.heart_disease_classification.train_model
```

Khởi động ứng dụng web:

```powershell
streamlit run src/heart_disease_classification/app.py
```

Nếu PowerShell chặn kích hoạt môi trường ảo, có thể chạy trực tiếp executable trong `.venv\Scripts\python.exe` và `.venv\Scripts\streamlit.exe`.

## 6. Cấu trúc thư mục

```text
heart-disease-classification/
├── data/
│   └── raw/processed.cleveland.data   # dữ liệu Cleveland gốc
├── models/heart_disease_pipeline.joblib
├── notebooks/heart_disease_classification.ipynb
├── reports/
│   ├── figures/                       # biểu đồ
│   ├── results/                       # CSV/JSON kết quả
│   ├── report.md
│   └── report.docx
├── src/
│   └── heart_disease_classification/
│       ├── app.py                       # ứng dụng Streamlit
│       └── train_model.py                # tiền xử lý, CV, đánh giá, lưu mô hình
├── requirements.txt
└── README.md
```

## 7. Giới hạn cần nêu trong báo cáo

- Dữ liệu Cleveland nhỏ (303 mẫu), cũ, có nguồn gốc từ một nhóm bệnh nhân cụ thể; không đại diện cho mọi quốc gia/nhóm dân số hiện tại.
- Mô hình chỉ phân lớp nhị phân, làm mất thông tin mức độ `num = 1..4` của nhãn gốc.
- Không có external validation, kiểm định tiến cứu, đánh giá hiệu chuẩn xác suất, phân tích sai lệch giữa các nhóm hay phê duyệt lâm sàng.
- Một số thuộc tính là mã phân loại; không diễn giải tương quan số học giữa mã phân loại như mối quan hệ tuyến tính thật.
- Kết quả app không thay thế hỏi bệnh, thăm khám, xét nghiệm hay ý kiến của bác sĩ.


## Bản cập nhật: phân tích đa cohort và trường quốc tịch

- Tab **Khám phá dữ liệu** hiển thị audit bốn tập processed Cleveland, Hungary, Switzerland và VA Long Beach.
- Form dự đoán có trường quốc tịch/nhóm tham chiếu. Trường này chỉ để ghi nhận bối cảnh, **không ảnh hưởng dự đoán** trong phiên bản hiện tại.
- Xem `reports/report.md` mục 10–12 và `reports/results/multi_cohort_dataset_audit.csv` để biết quy mô, tỷ lệ nhãn và mức thiếu dữ liệu từng tập.
- Mô hình chính vẫn dùng Cleveland; chưa được kiểm định đa quốc gia hoặc lâm sàng.
