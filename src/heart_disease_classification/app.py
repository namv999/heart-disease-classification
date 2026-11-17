"""Streamlit demo app for the UCI Cleveland Heart Disease classifier.

EDUCATIONAL PURPOSE ONLY: not for medical diagnosis or treatment decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
RESULTS_PATH = ROOT / "reports" / "results"
FIGURES_PATH = ROOT / "reports" / "figures"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

st.set_page_config(page_title="Heart Disease ML - Đồ án", page_icon="🫀", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có mô hình đã lưu. Chạy `python -m src.heart_disease_classification.train_model` tại thư mục gốc dự án trước."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    summary_path = RESULTS_PATH / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data
def load_comparison():
    p = RESULTS_PATH / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_raw_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")


st.title("🫀 Phân tích và phân lớp dữ liệu bệnh tim")
st.caption("Đồ án học máy minh họa sử dụng UCI Heart Disease - Cleveland subset")
st.warning(
    "**Chỉ phục vụ học tập và minh họa.** Đây không phải công cụ chẩn đoán y khoa, "
    "không được kiểm định lâm sàng và không dùng kết quả để tự chẩn đoán hay quyết định điều trị."
)

try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(str(exc))
    st.stop()

summary_tab, data_tab, model_tab, predict_tab = st.tabs(
    ["Tổng quan", "Khám phá dữ liệu", "Đánh giá mô hình", "Thử dự đoán minh họa"]
)

with summary_tab:
    st.subheader("Mục tiêu dự án")
    st.write(
        "Xây dựng quy trình phân lớp nhị phân từ các thuộc tính sức khỏe trong bộ dữ liệu Cleveland: "
        "nhãn 0 là không ghi nhận bệnh trong dữ liệu; nhãn 1 là có ghi nhận bệnh "
        "(gộp các giá trị nhãn gốc 1–4)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu", metadata.get("n_samples", "—"))
    c2.metric("Thuộc tính đầu vào", metadata.get("n_features", "—"))
    c3.metric("Mẫu train", metadata.get("train_samples", "—"))
    c4.metric("Mẫu test", metadata.get("test_samples", "—"))

    chosen = metadata.get("selected_model", "Chưa xác định")
    cvm = metadata.get("chosen_cv_metrics", {})
    testm = metadata.get("chosen_test_metrics", {})
    st.markdown(f"**Mô hình được chọn:** {chosen}")
    st.write("Quy tắc chọn: F1 trung bình cao nhất trong kiểm định chéo phân tầng 5-fold trên tập train.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Cross-validation trung bình trên tập train**")
        st.dataframe(pd.DataFrame([{
            "Accuracy": cvm.get("accuracy"), "Precision": cvm.get("precision"),
            "Recall": cvm.get("recall"), "F1": cvm.get("f1"), "ROC-AUC": cvm.get("roc_auc"),
        }]).round(3), hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Kết quả trên tập test giữ riêng**")
        st.dataframe(pd.DataFrame([{
            "Accuracy": testm.get("accuracy"), "Precision": testm.get("precision"),
            "Recall": testm.get("recall"), "F1": testm.get("f1"), "ROC-AUC": testm.get("roc_auc"),
        }]).round(3), hide_index=True, use_container_width=True)
    st.info(
        "Tập test chỉ có 61 mẫu, nên các chỉ số có thể dao động đáng kể theo cách chia dữ liệu. "
        "Không diễn giải điểm mô hình là xác suất mắc bệnh thực tế."
    )
    st.markdown("**Nguồn dữ liệu:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)")

with data_tab:
    st.subheader("Thông tin và chất lượng dữ liệu")
    raw = load_raw_data()
    if raw.empty:
        st.warning("Không tìm thấy tệp dữ liệu trong thư mục data.")
    else:
        pos = int(metadata.get("positive_samples", 0))
        neg = int(metadata.get("negative_samples", 0))
        a, b, c = st.columns(3)
        a.metric("Không ghi nhận bệnh", neg)
        b.metric("Có ghi nhận bệnh", pos)
        missing = metadata.get("missing_values", {})
        c.metric("Giá trị thiếu", sum(missing.values()))
        st.write("**Các cột:** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal; `num` là nhãn gốc.")
        st.write("`ca` có 4 giá trị thiếu và `thal` có 2 giá trị thiếu; pipeline dùng median cho biến số và mode cho biến phân loại.")
        st.markdown("**Xem một phần dữ liệu gốc**")
        st.dataframe(raw.head(12), hide_index=True, use_container_width=True)
        st.markdown("### Tổng quan các nhóm dữ liệu quốc tế trong gói UCI")
        audit_path = RESULTS_PATH / "multi_cohort_dataset_audit.csv"
        if audit_path.exists():
            audit_df = pd.read_csv(audit_path)
            st.dataframe(audit_df.rename(columns={
                "cohort": "Nhóm dữ liệu", "source_file": "Tệp nguồn", "n_rows": "Số mẫu",
                "positive_num_gt_0": "Nhãn dương (num > 0)", "negative_num_eq_0": "Nhãn âm (num = 0)",
                "positive_pct": "% nhãn dương", "missing_cells_predictors": "Ô thiếu ở đầu vào",
                "missing_predictor_pct": "% ô đầu vào thiếu", "rows_with_any_missing_predictor": "Mẫu có ít nhất 1 ô thiếu",
                "target_missing": "Nhãn thiếu", "age_median": "Tuổi trung vị"
            }), hide_index=True, use_container_width=True)
            st.caption("Các nhóm có tỷ lệ nhãn và mức thiếu dữ liệu rất khác nhau. Đây là mô tả dữ liệu, không phải so sánh nguy cơ bệnh giữa các quốc gia.")
        image_cols = st.columns(2)
        images = [
            ("target_distribution.png", "Phân bố nhãn"),
            ("missing_values.png", "Giá trị thiếu"),
            ("age_distribution.png", "Phân bố tuổi theo nhãn"),
            ("correlation_heatmap.png", "Ma trận tương quan"),
        ]
        for i, (filename, caption) in enumerate(images):
            image_path = FIGURES_PATH / filename
            with image_cols[i % 2]:
                if image_path.exists():
                    st.image(str(image_path), caption=caption, use_container_width=True)
        st.caption("Lưu ý: các trường dạng phân loại được lưu bằng mã số; tương quan Pearson giữa các mã số này chỉ mang tính thăm dò.")

with model_tab:
    st.subheader("So sánh thuật toán và kiểm tra mô hình")
    comparison = load_comparison()
    if not comparison.empty:
        display_cols = [
            "model", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1", "cv_roc_auc",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc",
        ]
        display_cols = [c for c in display_cols if c in comparison.columns]
        renamed = {
            "model": "Thuật toán", "cv_accuracy": "CV Accuracy", "cv_precision": "CV Precision",
            "cv_recall": "CV Recall", "cv_f1": "CV F1", "cv_roc_auc": "CV ROC-AUC",
            "test_accuracy": "Test Accuracy", "test_precision": "Test Precision",
            "test_recall": "Test Recall", "test_f1": "Test F1", "test_roc_auc": "Test ROC-AUC",
        }
        st.dataframe(comparison[display_cols].rename(columns=renamed).round(3), hide_index=True, use_container_width=True)
        st.markdown("**So sánh F1 qua 5-fold cross-validation**")
        chart_path = FIGURES_PATH / "algorithm_comparison.png"
        if chart_path.exists(): st.image(str(chart_path), use_container_width=True)
    cm_path = FIGURES_PATH / "confusion_matrix.png"
    roc_path = FIGURES_PATH / "roc_curve.png"
    c1, c2 = st.columns(2)
    with c1:
        if cm_path.exists(): st.image(str(cm_path), caption="Ma trận nhầm lẫn trên tập test", use_container_width=True)
    with c2:
        if roc_path.exists(): st.image(str(roc_path), caption="ROC curve trên tập test", use_container_width=True)
    st.markdown("**Cách đọc chỉ số**")
    st.write("- **Accuracy:** tỷ lệ dự đoán đúng trên tổng số mẫu.")
    st.write("- **Precision:** trong các mẫu mô hình dự đoán lớp 1, tỷ lệ mẫu thật sự thuộc lớp 1.")
    st.write("- **Recall/Sensitivity:** trong các mẫu thật sự thuộc lớp 1, tỷ lệ mô hình tìm ra; bỏ sót được phản ánh ở false negative.")
    st.write("- **F1:** trung bình điều hòa của precision và recall.")
    st.write("- **ROC-AUC:** khả năng xếp hạng mẫu dương cao hơn mẫu âm trên các ngưỡng phân loại; không chứng minh mô hình đã hiệu chuẩn xác suất.")
    coef_path = FIGURES_PATH / "logistic_coefficients.png"
    if coef_path.exists():
        st.image(str(coef_path), caption="Hệ số mô hình tuyến tính; không phải bằng chứng nhân quả", use_container_width=True)

with predict_tab:
    st.subheader("Nhập thuộc tính để xem kết quả phân lớp minh họa")
    st.write("Các giá trị bên dưới dùng mã hóa của bộ dữ liệu Cleveland. Chỉ nhập dữ liệu giả định hoặc dữ liệu đã được phép sử dụng.")
    with st.form("prediction_form"):
        nationality = st.selectbox(
            "Quốc tịch / nhóm tham chiếu (để ghi nhận bối cảnh)",
            ["Việt Nam", "Hoa Kỳ", "Hungary", "Thụy Sĩ", "Khác / không muốn cung cấp"],
            help="Hiện tại quốc tịch chỉ được ghi nhận trong giao diện; mô hình chưa được huấn luyện/kiểm định theo quốc tịch nên trường này không làm thay đổi điểm mô hình."
        )
        left, right = st.columns(2)
        with left:
            age = st.slider("Tuổi (age)", min_value=29, max_value=77, value=54)
            sex_label = st.selectbox("Giới tính (sex)", ["Nam (1)", "Nữ (0)"])
            cp_label = st.selectbox("Loại đau ngực (cp)", [
                "1 - Đau thắt ngực điển hình", "2 - Đau thắt ngực không điển hình",
                "3 - Đau không do thắt ngực", "4 - Không triệu chứng",
            ], index=1)
            trestbps = st.slider("Huyết áp lúc nghỉ (trestbps, mmHg)", 94, 200, 130)
            chol = st.slider("Cholesterol (chol, mg/dL)", 126, 564, 240)
            fbs_label = st.selectbox("Đường huyết lúc đói > 120 mg/dL (fbs)", ["Không (0)", "Có (1)"])
            restecg_label = st.selectbox("Điện tâm đồ lúc nghỉ (restecg)", [
                "0 - Bình thường", "1 - Bất thường ST-T", "2 - Phì đại thất trái theo tiêu chí dữ liệu",
            ])
        with right:
            thalach = st.slider("Nhịp tim tối đa đạt được (thalach)", 71, 202, 150)
            exang_label = st.selectbox("Đau thắt ngực do gắng sức (exang)", ["Không (0)", "Có (1)"])
            oldpeak = st.slider("ST chênh xuống do gắng sức (oldpeak)", 0.0, 6.2, 1.0, 0.1)
            slope_label = st.selectbox("Độ dốc đoạn ST (slope)", [
                "1 - Đi lên", "2 - Phẳng", "3 - Đi xuống",
            ])
            ca = st.selectbox("Số mạch chính được nhuộm màu (ca)", [0, 1, 2, 3])
            thal_label = st.selectbox("Kết quả thal (thal)", [
                "3 - Bình thường", "6 - Khiếm khuyết cố định", "7 - Khiếm khuyết có thể hồi phục",
            ])
        submitted = st.form_submit_button("Chạy mô hình minh họa", type="primary", use_container_width=True)

    if submitted:
        st.info(
            f"Nhóm tham chiếu đã chọn: **{nationality}**. Quốc tịch không được đưa vào mô hình dự đoán hiện tại; "
            "điểm số vẫn đến từ mô hình Cleveland. Không thể suy ra cơ địa hay nguy cơ cá nhân chỉ từ quốc tịch."
        )
        input_row = {
            "age": float(age),
            "sex": 1.0 if sex_label.startswith("Nam") else 0.0,
            "cp": float(int(cp_label.split(" ")[0])),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": 1.0 if fbs_label.startswith("Có") else 0.0,
            "restecg": float(int(restecg_label.split(" ")[0])),
            "thalach": float(thalach),
            "exang": 1.0 if exang_label.startswith("Có") else 0.0,
            "oldpeak": float(oldpeak),
            "slope": float(int(slope_label.split(" ")[0])),
            "ca": float(ca),
            "thal": float(int(thal_label.split(" ")[0])),
        }
        sample = pd.DataFrame([input_row])
        predicted_class = int(model.predict(sample)[0])
        class_probabilities = model.predict_proba(sample)[0]
        model_score = float(class_probabilities[list(model.classes_).index(1)])
        st.markdown("### Kết quả của mô hình")
        if predicted_class == 1:
            st.error("Mô hình xếp mẫu này vào lớp 1: *có ghi nhận bệnh* trong dữ liệu huấn luyện.")
        else:
            st.success("Mô hình xếp mẫu này vào lớp 0: *không ghi nhận bệnh* trong dữ liệu huấn luyện.")
        st.metric("Điểm mô hình cho lớp 1 (không phải xác suất lâm sàng)", f"{model_score:.1%}")
        st.caption(
            "Kết quả chỉ minh họa cách gọi pipeline đã huấn luyện. Giá trị đầu vào, đặc điểm quần thể, "
            "thiết bị xét nghiệm và quy trình thu thập dữ liệu có thể khác thực tế. Không dùng kết quả này "
            "để xác nhận hoặc loại trừ bệnh; hãy trao đổi với nhân viên y tế có chuyên môn."
        )
        with st.expander("Xem dữ liệu đầu vào sau khi mã hóa"):
            st.dataframe(sample, hide_index=True, use_container_width=True)

st.divider()
st.caption("Nguồn: UCI Machine Learning Repository. Mô hình được huấn luyện lại bằng `src/heart_disease_classification/train_model.py`; dùng `streamlit run src/heart_disease_classification/app.py` để mở ứng dụng cục bộ.")




