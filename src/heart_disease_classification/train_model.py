"""Train and evaluate classifiers for the UCI Cleveland Heart Disease dataset.

Educational/research demonstration only. Not a medical diagnostic system.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
REPORTS_PATH = ROOT / "reports"
RESULTS_PATH = REPORTS_PATH / "results"
FIGURES_PATH = REPORTS_PATH / "figures"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    c for c in COLUMNS if c not in NUMERIC_FEATURES and c != "num"
]
FEATURE_DESCRIPTIONS = {
    "age": "Tuổi (năm)", "sex": "Giới tính (1=nam, 0=nữ)",
    "cp": "Loại đau ngực (1-4)", "trestbps": "Huyết áp lúc nghỉ (mmHg)",
    "chol": "Cholesterol huyết thanh (mg/dL)",
    "fbs": "Đường huyết lúc đói > 120 mg/dL (1=có)",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ (0-2)",
    "thalach": "Nhịp tim tối đa đạt được", "exang": "Đau thắt ngực do gắng sức (1=có)",
    "oldpeak": "ST chênh xuống khi gắng sức", "slope": "Độ dốc đoạn ST đỉnh (1-3)",
    "ca": "Số mạch chính được nhuộm màu (0-3)",
    "thal": "Kết quả thal (3/6/7)",
}


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")
    for col in COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    # Target gốc: 0, 1, 2, 3, 4; chuyển về nhị phân theo thiết lập thí nghiệm UCI.
    y = (raw["num"] > 0).astype(int).rename("target")
    X = raw.drop(columns="num")
    return X, y, raw


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=4, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    return {name: Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
            for name, model in algorithms.items()}


def evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["Không ghi nhận bệnh", "Có ghi nhận bệnh"],
            output_dict=True, zero_division=0,
        ),
    }


def create_figures(X: pd.DataFrame, y: pd.Series, raw: pd.DataFrame,
                   candidate_results: pd.DataFrame, chosen_name: str,
                   test_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, str]:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Class balance.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    counts = y.map({0: "Không ghi nhận bệnh (0)", 1: "Có ghi nhận bệnh (1)"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, color="steelblue", ax=ax)
    ax.set_title("Phân bố nhãn mục tiêu")
    ax.set_xlabel("")
    ax.set_ylabel("Số mẫu")
    for i, v in enumerate(counts.values): ax.text(i, v + 2, str(v), ha="center")
    fig.tight_layout()
    paths["target_distribution"] = str(FIGURES_PATH / "target_distribution.png")
    fig.savefig(paths["target_distribution"], dpi=160)
    plt.close(fig)

    # 2. Missing values.
    missing = X.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    miss = missing[missing > 0]
    if len(miss):
        sns.barplot(x=miss.index, y=miss.values, color="steelblue", ax=ax)
        ax.set_ylabel("Số giá trị thiếu")
        ax.set_xlabel("Thuộc tính")
        ax.set_title("Giá trị thiếu trước tiền xử lý")
        for i, v in enumerate(miss.values): ax.text(i, v + 0.05, str(int(v)), ha="center")
    else:
        ax.text(.5, .5, "Không có giá trị thiếu", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    paths["missing_values"] = str(FIGURES_PATH / "missing_values.png")
    fig.savefig(paths["missing_values"], dpi=160)
    plt.close(fig)

    # 3. Age distribution by target.
    plot_df = X[["age"]].copy()
    plot_df["Nhãn"] = y.map({0: "Không ghi nhận bệnh", 1: "Có ghi nhận bệnh"})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.histplot(data=plot_df, x="age", hue="Nhãn", multiple="layer", bins=18,
                 element="step", stat="count", common_norm=False, ax=ax)
    ax.set_title("Phân bố tuổi theo nhãn trong bộ dữ liệu")
    ax.set_xlabel("Tuổi (năm)")
    ax.set_ylabel("Số mẫu")
    fig.tight_layout()
    paths["age_distribution"] = str(FIGURES_PATH / "age_distribution.png")
    fig.savefig(paths["age_distribution"], dpi=160)
    plt.close(fig)

    # 4. Exploratory correlation heatmap (coded categories are numeric codes; interpret cautiously).
    corr_df = X.copy()
    corr_df["target"] = y
    corr = corr_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, square=True,
                linewidths=.35, cbar_kws={"shrink": .75}, ax=ax)
    ax.set_title("Ma trận tương quan (mã danh mục là mã số, cần diễn giải thận trọng)")
    fig.tight_layout()
    paths["correlation_heatmap"] = str(FIGURES_PATH / "correlation_heatmap.png")
    fig.savefig(paths["correlation_heatmap"], dpi=160)
    plt.close(fig)

    # 5. Algorithm comparison using mean cross-validation metrics.
    fig, ax = plt.subplots(figsize=(10.3, 5.6))
    ordered = candidate_results.sort_values("cv_f1", ascending=False)
    sns.barplot(data=ordered, x="cv_f1", y="model", color="steelblue", ax=ax)
    ax.set_title("So sánh thuật toán theo F1 trung bình (5-fold CV, tập train)")
    ax.set_xlabel("F1 trung bình")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    paths["algorithm_comparison"] = str(FIGURES_PATH / "algorithm_comparison.png")
    fig.savefig(paths["algorithm_comparison"], dpi=160)
    plt.close(fig)

    # 6. Holdout confusion matrix and ROC.
    pred = test_model.predict(X_test)
    prob = test_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Dự đoán 0", "Dự đoán 1"],
                yticklabels=["Thực tế 0", "Thực tế 1"], ax=ax)
    ax.set_title(f"Ma trận nhầm lẫn - {chosen_name} (tập test)")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    fig.tight_layout()
    paths["confusion_matrix"] = str(FIGURES_PATH / "confusion_matrix.png")
    fig.savefig(paths["confusion_matrix"], dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, prob)
    test_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.plot(fpr, tpr, label=f"{chosen_name} (AUC = {test_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Đường ngẫu nhiên")
    ax.set_title("ROC trên tập kiểm tra độc lập")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Sensitivity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    paths["roc_curve"] = str(FIGURES_PATH / "roc_curve.png")
    fig.savefig(paths["roc_curve"], dpi=160)
    plt.close(fig)

    # 7. Logistic coefficients, only when the selected model is logistic regression.
    if chosen_name == "Logistic Regression":
        prep = test_model.named_steps["preprocess"]
        feat_names = prep.get_feature_names_out()
        coefs = test_model.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame({"feature": feat_names, "coefficient": coefs})
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.nlargest(15, "abs_coefficient").sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ax.barh(coef_df["feature"], coef_df["coefficient"])
        ax.axvline(0, linewidth=1)
        ax.set_title("15 hệ số Logistic Regression có trị tuyệt đối lớn nhất")
        ax.set_xlabel("Hệ số (sau chuẩn hóa/one-hot); không phải quan hệ nhân quả")
        ax.set_ylabel("Đặc trưng đã biến đổi")
        fig.tight_layout()
        paths["logistic_coefficients"] = str(FIGURES_PATH / "logistic_coefficients.png")
        fig.savefig(paths["logistic_coefficients"], dpi=160)
        plt.close(fig)
        coef_df.to_csv(RESULTS_PATH / "logistic_coefficients.csv", index=False)
    return paths


def main() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    X, y, raw = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = make_preprocessor()
    candidates = build_candidates(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc",
    }

    rows = []
    fitted_train: dict[str, Pipeline] = {}
    holdout_by_model = {}
    for name, pipeline in candidates.items():
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, error_score="raise"
        )
        pipeline.fit(X_train, y_train)
        fitted_train[name] = pipeline
        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            vals = cv_results[f"test_{metric}"]
            row[f"cv_{metric}"] = float(np.mean(vals))
            row[f"cv_{metric}_std"] = float(np.std(vals, ddof=1))
        # Holdout metrics are included for transparency, but the winner is selected ONLY by CV F1.
        test_predictions = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        row.update({
            "test_accuracy": float(accuracy_score(y_test, test_predictions)),
            "test_precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        })
        rows.append(row)
        holdout_by_model[name] = evaluate_holdout(pipeline, X_test, y_test)

    comparison = pd.DataFrame(rows).sort_values(
        ["cv_f1", "cv_roc_auc", "cv_recall"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(RESULTS_PATH / "model_comparison.csv", index=False)

    chosen_name = str(comparison.iloc[0]["model"])
    chosen_for_test = fitted_train[chosen_name]
    chosen_test_metrics = holdout_by_model[chosen_name]
    y_pred = chosen_for_test.predict(X_test)

    # Refit the chosen pipeline on all available records for the app artifact after holdout evaluation.
    final_model = clone(candidates[chosen_name])
    final_model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    figures = create_figures(X, y, raw, comparison, chosen_name,
                             chosen_for_test, X_test, y_test)

    positive = int(y.sum())
    target_counts = {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()}
    missing_counts = {str(k): int(v) for k, v in X.isna().sum().items() if int(v) > 0}
    metadata = {
        "dataset_name": "UCI Heart Disease - processed Cleveland subset",
        "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "positive_samples": positive, "negative_samples": int(len(y) - positive),
        "target_counts": target_counts, "missing_values": missing_counts,
        "target_definition": "0 = num == 0; 1 = num > 0 (gộp các mức gốc 1-4 thành lớp có ghi nhận bệnh)",
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "split": "train_test_split(test_size=0.20, stratify=y, random_state=42)",
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on training set",
        "selection_rule": "Chọn mô hình có F1 trung bình 5-fold CV cao nhất trên tập train; không chọn theo kết quả test.",
        "selected_model": chosen_name,
        "chosen_cv_metrics": {k: float(comparison.iloc[0][f"cv_{k}"]) for k in scoring},
        "chosen_test_metrics": {k: float(v) for k, v in chosen_test_metrics.items()
                                if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "chosen_test_confusion_matrix": chosen_test_metrics["confusion_matrix"],
        "chosen_test_classification_report": chosen_test_metrics["classification_report"],
        "test_class_counts": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "figures": {k: str(Path(v).relative_to(ROOT)) for k, v in figures.items()},
        "feature_descriptions_vi": FEATURE_DESCRIPTIONS,
        "educational_notice": "Chỉ phục vụ học tập/minh họa; không phải thiết bị hoặc công cụ chẩn đoán y khoa. Không sử dụng cho quyết định điều trị.",
    }
    (RESULTS_PATH / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Save holdout predictions to make test results reproducible and inspectable.
    prediction_df = X_test.copy()
    prediction_df["actual_target"] = y_test
    prediction_df["predicted_target"] = y_pred
    prediction_df["model_score"] = chosen_for_test.predict_proba(X_test)[:, 1]
    prediction_df.to_csv(RESULTS_PATH / "holdout_predictions.csv", index=False)

    print("=" * 72)
    print("UCI HEART DISEASE - TRAINING COMPLETE")
    print(f"Samples/features: {len(X)}/{X.shape[1]} | Positive samples: {positive} ({positive/len(y):.1%})")
    print(f"Missing values: {missing_counts}")
    print(f"Selected model by 5-fold CV F1: {chosen_name}")
    print(f"CV metrics: {metadata['chosen_cv_metrics']}")
    print(f"Holdout test metrics: {metadata['chosen_test_metrics']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {chosen_test_metrics['confusion_matrix']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Reports: {REPORTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()


