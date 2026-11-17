# Báo cáo đồ án: Phân tích và phân lớp dữ liệu bệnh tim bằng Machine Learning

**Phạm vi:** UCI Heart Disease - Cleveland processed subset  
**Mục tiêu:** phân tích dữ liệu, trực quan hóa, so sánh các thuật toán phân lớp, huấn luyện, kiểm tra, đánh giá, lưu pipeline và xây dựng app minh họa.

> **Giới hạn an toàn:** Đây là đồ án học máy trên bộ dữ liệu công khai nhỏ. Ứng dụng không phải thiết bị y tế, không được xác thực lâm sàng và tuyệt đối không dùng để tự chẩn đoán, loại trừ bệnh hay quyết định điều trị.

## 1. Bài toán và nguồn dữ liệu

Bộ dữ liệu Heart Disease được lưu trữ tại UCI Machine Learning Repository. Trang nguồn mô tả nhiều tập dữ liệu liên quan đến bệnh tim; Cleveland là một tập được sử dụng phổ biến trong các bài toán học máy. Phiên bản processed Cleveland trong ZIP có **303 mẫu, 13 thuộc tính đầu vào và một thuộc tính mục tiêu**. Nhãn gốc `num` nhận các giá trị 0 đến 4; theo cách thiết lập phân lớp phổ biến trên tập này, chuyển mục tiêu thành nhị phân: `num = 0` thành lớp 0 và `num > 0` thành lớp 1 [1].

### 1.1. Các thuộc tính đầu vào

| Thuộc tính | Ý nghĩa | Kiểu sử dụng trong pipeline |
|---|---|---|
| `age` | Tuổi (năm) | Liên tục |
| `sex` | Giới tính được mã hóa 0/1 | Phân loại |
| `cp` | Loại đau ngực, mã 1-4 | Phân loại |
| `trestbps` | Huyết áp lúc nghỉ (mmHg) | Liên tục |
| `chol` | Cholesterol huyết thanh (mg/dL) | Liên tục |
| `fbs` | Đường huyết lúc đói > 120 mg/dL | Phân loại 0/1 |
| `restecg` | Kết quả ECG lúc nghỉ, mã 0-2 | Phân loại |
| `thalach` | Nhịp tim tối đa đạt được | Liên tục |
| `exang` | Đau thắt ngực do gắng sức, 0/1 | Phân loại |
| `oldpeak` | Độ chênh ST do gắng sức so với lúc nghỉ | Liên tục |
| `slope` | Độ dốc đoạn ST tại đỉnh gắng sức, mã 1-3 | Phân loại |
| `ca` | Số mạch chính nhuộm màu bằng fluoroscopy, mã 0-3 | Phân loại |
| `thal` | Mã phân loại kết quả thal (3/6/7) | Phân loại |

Các cách mã hóa biến được đối chiếu theo tài liệu thuộc tính UCI [2]. Đây là mã hóa của bộ dữ liệu nghiên cứu, không phải hướng dẫn để người dùng tự diễn giải kết quả xét nghiệm.

### 1.2. Phân bố nhãn và dữ liệu thiếu

| Nội dung | Kết quả |
|---|---:|
| Tổng số mẫu | 303 |
| Lớp 0 - `num = 0` | 164 (54,1%) |
| Lớp 1 - `num > 0` | 139 (45,9%) |
| Giá trị thiếu ở `ca` | 4 |
| Giá trị thiếu ở `thal` | 2 |

Không thấy chỉ báo thiếu ở các cột đầu vào khác. Tệp có tổng cộng 6 giá trị thiếu trên 303 mẫu. Mất cân bằng hai lớp không quá lớn, tuy vậy vẫn cần xem đồng thời precision và recall thay vì chỉ dùng accuracy.

## 2. Tìm hiểu các thuật toán phân lớp

Phân lớp có giám sát học mối quan hệ giữa thuộc tính đầu vào `X` và nhãn rời rạc `y`. Dự án đánh giá 8 ứng viên:

1. **Logistic Regression:** mô hình tuyến tính ước lượng xác suất thuộc lớp dương qua hàm logistic; cách cài đặt của scikit-learn hỗ trợ phân lớp nhị phân và đa lớp. Ưu điểm tương đối dễ giải thích, thời gian huấn luyện nhanh; hạn chế là ranh giới mặc định tuyến tính theo đặc trưng đã biến đổi [3].
2. **Decision Tree:** học các quy tắc phân nhánh trên thuộc tính. Dễ minh họa, nhưng cây sâu có thể khớp quá mức với tập huấn luyện [4].
3. **Random Forest:** kết hợp nhiều cây để giảm phụ thuộc vào một cây đơn lẻ, tạo mô hình phi tuyến [5].
4. **Extra Trees:** cũng là phương pháp tổ hợp cây, bổ sung ngẫu nhiên trong quá trình tạo cây; dùng để so sánh với Random Forest [5].
5. **SVM (RBF):** xây dựng ranh giới phân lớp dựa vào support vectors; kernel RBF cho phép ranh giới phi tuyến. Cần chuẩn hóa thuộc tính số; SVM cũng cần lựa chọn siêu tham số phù hợp [6].
6. **K-Nearest Neighbors (KNN):** gán nhãn theo các mẫu huấn luyện gần nhất; nhạy với thang đo và tham số `k`, vì thế dùng StandardScaler cho biến số [7].
7. **Gaussian Naive Bayes:** mô hình xác suất dựa trên giả định độc lập có điều kiện giữa các đặc trưng; gọn nhẹ nhưng giả định này có thể không phản ánh cấu trúc dữ liệu thật [8].
8. **Gradient Boosting:** huấn luyện tuần tự các mô hình yếu, ở đây là các cây, để cải thiện sai số dự đoán [5].

Không thể xác định thuật toán tốt nhất một cách chung cho mọi dữ liệu. Do đó, dự án dùng cùng tập train, cùng quy trình xử lý dữ liệu và cùng các thước đo để so sánh thực nghiệm.

## 3. Tiền xử lý và thiết kế thực nghiệm

### 3.1. Làm sạch dữ liệu

- Đọc `?` thành giá trị thiếu, chuyển kiểu dữ liệu về số.
- Tách `num` khỏi X, sau đó tạo `y = 1` nếu `num > 0`, ngược lại `y = 0`.
- Không xóa 6 dòng có giá trị thiếu. Median imputation được dùng cho biến liên tục; most-frequent imputation cho biến phân loại.
- Với biến liên tục (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`), dùng `StandardScaler` để đưa về thang đo phù hợp cho Logistic Regression, SVM và KNN.
- Với biến phân loại, dùng `OneHotEncoder(handle_unknown="ignore")`; điều này tránh áp đặt thứ tự số học sai lên các mã danh mục.
- Toàn bộ các bước tiền xử lý đặt trong `Pipeline`/`ColumnTransformer`, vì vậy imputer và scaler được fit bên trong từng fold, hạn chế rò rỉ dữ liệu.

### 3.2. Chia dữ liệu và chọn mô hình

- Tập huấn luyện: 80% (242 mẫu); tập kiểm tra giữ riêng: 20% (61 mẫu).
- Chia bằng `train_test_split`, `stratify=y`, `random_state=42` để giữ gần đúng tỷ lệ hai lớp và tái lập kết quả.
- Trên tập train, chạy `StratifiedKFold` với 5 fold; ghi nhận mean và độ lệch chuẩn cho Accuracy, Precision, Recall, F1 và ROC-AUC.
- Chọn mô hình có **F1 trung bình cross-validation cao nhất**; tie-break theo ROC-AUC rồi Recall. Tập test không được dùng làm tiêu chí chọn mô hình.
- Sau đánh giá holdout, mô hình đã chọn được fit lại trên toàn bộ dữ liệu để tạo file phục vụ ứng dụng. Các chỉ số holdout dưới đây là từ mô hình chỉ fit trên tập train trước khi refit, không phải hiệu suất được đo trên dữ liệu dùng để fit artifact cuối cùng.

### 3.3. Các thước đo đánh giá

- **Accuracy:** tỷ lệ mẫu được phân lớp đúng.
- **Precision:** tỷ lệ dự đoán dương tính đúng trong các mẫu mô hình dự đoán dương.
- **Recall/Sensitivity:** tỷ lệ mẫu dương thật sự mà mô hình phát hiện; recall thấp có nghĩa có thể bỏ sót nhiều mẫu dương.
- **F1:** trung bình điều hòa precision và recall, phù hợp khi cần cân bằng hai thước đo này [9].
- **ROC-AUC:** đo khả năng xếp hạng mẫu dương cao hơn mẫu âm khi thay đổi ngưỡng; chỉ số này không tự chứng minh xác suất đã hiệu chuẩn tốt [10].
- **Confusion matrix:** cho biết TN, FP, FN và TP để nhìn thấy loại lỗi thay vì chỉ xem một con số tổng hợp.

## 4. Phân tích khám phá và trực quan hóa

### 4.1. Thống kê mô tả các biến số

| Biến | Trung bình | Độ lệch chuẩn | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Tuổi (`age`) | 54,44 | 9,04 | 29 | 56 | 77 |
| Huyết áp nghỉ (`trestbps`) | 131,69 | 17,60 | 94 | 130 | 200 |
| Cholesterol (`chol`) | 246,69 | 51,78 | 126 | 241 | 564 |
| Nhịp tim tối đa (`thalach`) | 149,61 | 22,88 | 71 | 153 | 202 |
| `oldpeak` | 1,04 | 1,16 | 0,00 | 0,80 | 6,20 |

### 4.2. Quan sát mô tả theo nhãn

| Biến trung bình | Lớp 0 | Lớp 1 |
|---|---:|---:|
| Tuổi | 52,59 | 56,63 |
| Huyết áp nghỉ | 129,25 | 134,57 |
| Cholesterol | 242,64 | 251,47 |
| Nhịp tim tối đa | 158,38 | 139,26 |
| `oldpeak` | 0,59 | 1,57 |

Trong tập Cleveland này, lớp 1 có tuổi trung bình và `oldpeak` trung bình cao hơn, trong khi `thalach` trung bình thấp hơn lớp 0. Đây chỉ là khác biệt mô tả trong một bộ dữ liệu lịch sử; không được diễn giải thành quan hệ nhân quả hoặc quy tắc y khoa. Mã danh mục như `cp`, `restecg`, `slope`, `ca`, `thal` được trực quan hóa bằng tần suất hoặc One-Hot Encoding; tương quan trên mã số danh mục chỉ là tham khảo.

Biểu đồ được sinh tự động trong `reports/figures/`:

- `target_distribution.png`: số lượng mỗi lớp.
- `missing_values.png`: số lượng giá trị thiếu theo cột.
- `age_distribution.png`: phân bố tuổi theo nhãn.
- `correlation_heatmap.png`: ma trận tương quan khám phá.
- `algorithm_comparison.png`: so sánh F1 CV giữa các thuật toán.
- `confusion_matrix.png` và `roc_curve.png`: kiểm tra mô hình trên tập test.
- `logistic_coefficients.png`: hệ số của Logistic Regression sau tiền xử lý; không mang ý nghĩa quan hệ nhân quả.

## 5. Kết quả so sánh mô hình

### 5.1. Cross-validation trên tập train

| Thuật toán | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,855 | 0,873 | 0,810 | **0,837** | 0,902 |
| Extra Trees | 0,826 | 0,809 | 0,819 | 0,810 | **0,904** |
| SVM (RBF) | 0,822 | 0,826 | 0,792 | 0,805 | 0,888 |
| K-Nearest Neighbors | 0,818 | 0,827 | 0,766 | 0,794 | 0,872 |
| Random Forest | 0,806 | 0,797 | 0,783 | 0,784 | 0,897 |
| Gradient Boosting | 0,798 | 0,772 | 0,792 | 0,781 | 0,857 |
| Gaussian Naive Bayes | 0,744 | 0,705 | 0,846 | 0,749 | 0,877 |
| Decision Tree | 0,694 | 0,663 | 0,693 | 0,675 | 0,795 |

**Lý do chọn Logistic Regression:** F1 cross-validation cao nhất (0,837), Accuracy CV cũng cao nhất (0,855), trong khi ROC-AUC 0,902 gần mức cao nhất của Extra Trees (0,904). Với dữ liệu nhỏ, ưu điểm tương đối dễ diễn giải và ít phức tạp hơn là lý do bổ sung để ưu tiên mô hình này cho bản demo. Đây không khẳng định nó luôn tốt hơn các mô hình khác trên dữ liệu mới.

### 5.2. Đánh giá trên tập test 61 mẫu

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | **0,869 (86,9%)** |
| Precision | **0,813 (81,3%)** |
| Recall/Sensitivity | **0,929 (92,9%)** |
| F1 | **0,867 (86,7%)** |
| ROC-AUC | **0,966** |

Confusion matrix, hàng là nhãn thật và cột là nhãn dự đoán (`[[TN, FP], [FN, TP]]`):

| | Dự đoán 0 | Dự đoán 1 |
|---|---:|---:|
| Thực tế 0 | 27 | 6 |
| Thực tế 1 | 2 | 26 |

Trong 61 mẫu test, mô hình phân loại đúng 53 mẫu, gồm 27 mẫu lớp 0 và 26 mẫu lớp 1; nhầm 6 mẫu lớp 0 thành lớp 1 và bỏ sót 2 mẫu lớp 1. Do test set nhỏ, thay đổi vài mẫu có thể làm các chỉ số biến động mạnh. Recall cao ở lần chia này không đảm bảo hiệu năng tương tự ở quần thể khác.

## 6. Lưu mô hình và xây dựng ứng dụng

### 6.1. Mô hình lưu

Artifact `models/heart_disease_pipeline.joblib` là toàn bộ pipeline gồm imputer, scaler, One-Hot Encoder và Logistic Regression đã fit lại trên 303 dòng. Lưu cả pipeline giúp app áp dụng cùng phép biến đổi như lúc huấn luyện. Việc tải file joblib chỉ nên thực hiện với artifact tin cậy do thư viện pickle/joblib có rủi ro thực thi mã khi nạp file không tin cậy.

### 6.2. App minh họa

`src/heart_disease_classification/app.py` sử dụng Streamlit, tạo 4 khu vực:

1. **Tổng quan:** quy mô dữ liệu, mô hình đã chọn, metric cross-validation và test.
2. **Khám phá dữ liệu:** xem một phần dữ liệu và biểu đồ EDA.
3. **Đánh giá mô hình:** bảng so sánh 8 thuật toán, confusion matrix, ROC curve, hệ số mô hình.
4. **Thử dự đoán minh họa:** nhập 13 thuộc tính theo mã hóa của bộ dữ liệu và gọi `predict`/`predict_proba` từ pipeline đã lưu.

Streamlit hỗ trợ tạo ứng dụng dữ liệu tương tác bằng Python và chạy cục bộ bằng lệnh `streamlit run src/heart_disease_classification/app.py` [11].

### 6.3. Chạy ứng dụng

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.heart_disease_classification.train_model
streamlit run src/heart_disease_classification/app.py
```

Nếu chỉ muốn mở app dùng artifact đã đóng gói, có thể bỏ qua `python -m src.heart_disease_classification.train_model` miễn là file trong thư mục `models/` và `reports/results/` còn nguyên.

## 7. Kết luận và hướng cải thiện

Dự án đã xây dựng quy trình machine learning có thể tái lập từ dữ liệu đến app. Trong lần chạy hiện tại, Logistic Regression đứng đầu theo F1 trung bình qua 5-fold CV; trên holdout test, mô hình đạt accuracy 86,9% và recall 92,9%. Các giá trị này chỉ đánh giá khả năng phân lớp trên cách chia của bộ dữ liệu, không phải xác nhận mô hình có giá trị lâm sàng.

Hướng cải thiện phù hợp nếu tiếp tục phát triển:

- Thu thập hoặc dùng bộ dữ liệu độc lập mới để external validation.
- Tối ưu siêu tham số chỉ trong cross-validation, ví dụ `GridSearchCV` lồng trong quy trình train.
- Phân tích precision-recall curve và quyết định ngưỡng theo chi phí false negative/false positive đã được chuyên gia xác định.
- Kiểm tra calibration, sensitivity/specificity theo nhóm, khoảng tin cậy, và khả năng tổng quát giữa các cơ sở y tế.
- Bắt buộc có thẩm định lâm sàng, quản trị dữ liệu và phê duyệt chuyên môn trước bất kỳ ý định sử dụng thực tế nào.

## Tài liệu tham khảo

1. UCI Machine Learning Repository, **Heart Disease**. https://archive.ics.uci.edu/dataset/45/heart+disease
2. UCI Machine Learning Repository, **Heart Disease - Additional Variable Information**. https://archive.ics.uci.edu/dataset/45/heart+disease
3. scikit-learn, **Logistic Regression - Linear Models user guide**. https://scikit-learn.org/stable/modules/linear_model.html
4. scikit-learn, **Decision Trees user guide**. https://scikit-learn.org/stable/modules/tree.html
5. scikit-learn, **Ensemble methods user guide**. https://scikit-learn.org/stable/modules/ensemble.html
6. scikit-learn, **Support Vector Machines user guide**. https://scikit-learn.org/stable/modules/svm.html
7. scikit-learn, **Nearest Neighbors user guide**. https://scikit-learn.org/stable/modules/neighbors.html
8. scikit-learn, **Naive Bayes user guide**. https://scikit-learn.org/stable/modules/naive_bayes.html
9. scikit-learn, **f1_score documentation**. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
10. scikit-learn, **roc_auc_score documentation**. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
11. Streamlit, **Get started / Create an app**. https://docs.streamlit.io/get-started/tutorials/create-an-app


## 10. Mở rộng phân tích: sử dụng các tập dữ liệu processed trong gói UCI

Ngoài Cleveland, gói UCI được cung cấp còn có các tập processed Hungary, Switzerland và VA Long Beach. Báo cáo đã bổ sung `reports/results/multi_cohort_dataset_audit.csv` và `.json` để tổng hợp toàn bộ bốn nhóm có cùng cấu trúc 14 cột. Quy ước nhãn mô tả ở bảng là `num > 0` (dương) so với `num = 0` (âm), tương tự thiết lập nhị phân của thí nghiệm Cleveland.

| Nhóm dữ liệu | Số mẫu | num > 0 | num = 0 | Tỷ lệ num > 0 | Ô thiếu ở 13 thuộc tính đầu vào |
|---|---:|---:|---:|---:|---:|
| United States – Cleveland | 303 | 139 | 164 | 45.87% | 6 (0.15%) |
| Hungary | 294 | 106 | 188 | 36.05% | 782 (20.46%) |
| Switzerland | 123 | 115 | 8 | 93.50% | 273 (17.07%) |
| United States – VA Long Beach | 200 | 149 | 51 | 74.50% | 698 (26.85%) |

Tổng cộng bốn tập có **920 mẫu**. Tỷ lệ `num > 0` khác biệt đáng kể giữa các tập: Cleveland 45.87%, Hungary 36.05%, Switzerland 93.50%, VA Long Beach 74.50%. Đây là tỷ lệ nhãn trong các tập được thu thập, **không phải** tỷ lệ hiện mắc đại diện cho dân số từng quốc gia. Các tập cũng khác nhau về độ thiếu dữ liệu; Hungary có 20.46%, Switzerland 17.07% và VA Long Beach 26.85% ô đầu vào bị thiếu. Điều này cảnh báo không nên gộp dữ liệu thô rồi diễn giải kết quả như một quần thể đồng nhất.

### Vì sao mô hình dự đoán chính vẫn dựa trên Cleveland?

Mô hình và các chỉ số đánh giá chính vẫn giữ nguyên thiết lập Cleveland để duy trì tính tái lập và tránh đưa ra tuyên bố hiệu năng mới khi chưa thiết kế/kiểm định thí nghiệm đa tập phù hợp. Phân tích bốn tập hiện được dùng cho **tổng quan dữ liệu**, chưa phải một mô hình đã được huấn luyện và xác thực đa quốc gia. Trộn các tập có thể gây rò rỉ nguồn/quần thể, lệch phân phối, thiếu dữ liệu không đồng đều và khác biệt quy trình đo lường. Bước tiếp theo đúng về phương pháp là đánh giá ngoài mẫu theo từng cohort (train trên cohort này, test trên cohort khác), báo cáo khoảng tin cậy và hiệu chuẩn, thay vì chỉ báo cáo một điểm số gộp.

## 11. Quốc tịch trong giao diện: có ghi nhận nhưng không dùng như đại diện cho cơ địa

Form dự đoán đã bổ sung trường **Quốc tịch / nhóm tham chiếu** để ghi nhận bối cảnh người dùng. Trong phiên bản hiện tại, trường này **không được đưa vào pipeline và không làm thay đổi kết quả mô hình**. Lý do: bộ dữ liệu không có trường quốc tịch đáng tin cậy cho từng cá nhân và bốn cohort không đại diện đầy đủ cho các quốc tịch/dân số. Gán quốc tịch từ nguồn dữ liệu cũng không đồng nghĩa quốc tịch là nguyên nhân sinh học.

Để nghiên cứu khác biệt giữa quần thể một cách có trách nhiệm, cần dữ liệu đại diện và được phép sử dụng, định nghĩa rõ biến dân số/nguồn gốc, đánh giá riêng độ nhạy, độ đặc hiệu, calibration và sai số theo nhóm, đồng thời có kiểm định lâm sàng. Không nên thay đổi ngưỡng hay đưa ra lời khuyên cá nhân chỉ dựa trên quốc tịch.

## 12. Những gì đã cập nhật trong phiên bản này

- Thêm bảng audit dữ liệu bốn cohort vào tab **Khám phá dữ liệu** của Streamlit.
- Thêm lựa chọn quốc tịch/nhóm tham chiếu vào form; giao diện giải thích rõ trường này chỉ là metadata, không ảnh hưởng điểm mô hình.
- Thêm `reports/results/multi_cohort_dataset_audit.csv` và `reports/results/multi_cohort_dataset_audit.json`.
- Giữ nguyên mô hình Cleveland và các metric hiện có; không bịa thêm kết quả đánh giá đa quốc gia.
- Các giới hạn y tế được nhắc lại: điểm mô hình không phải xác suất lâm sàng, và app chỉ dùng cho học tập/minh họa.




