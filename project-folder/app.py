import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Heart Health Analytics", layout="wide")
st.title("❤️ Heart Health Analytics")
st.write("Ajay Gautham J – 2025aa05324 – ML Assignment 02")

# --------------------------------------------------
# GitHub RAW Dataset URL
# --------------------------------------------------
GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/ajaygautham05/"
    "2025aa05324-bits-aiml/main/project-folder/heart.csv"
)

# --------------------------------------------------
# Dataset Download Section
# --------------------------------------------------
st.subheader("📥 Download Dataset (GitHub Source)")

try:
    csv_text = pd.read_csv(GITHUB_DATA_URL).to_csv(index=False)

    st.download_button(
        label="⬇️ Download Dataset (heart.csv)",
        data=csv_text,
        file_name="heart.csv",
        mime="text/csv"
    )

    if st.button("📊 Load GitHub Dataset for Analysis"):
        st.session_state["data"] = pd.read_csv(StringIO(csv_text))
        st.success("Dataset loaded successfully from GitHub!")

except Exception:
    st.error("Unable to fetch dataset from GitHub.")
    st.stop()

# --------------------------------------------------
# Dataset Upload Section (NEW FEATURE)
# --------------------------------------------------
st.subheader("📤 Upload Your Own Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a `target` column",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if "target" not in uploaded_df.columns:
            st.error("Uploaded dataset must contain a 'target' column.")
            st.stop()

        st.session_state["data"] = uploaded_df
        st.success("Uploaded dataset loaded successfully!")

    except Exception as e:
        st.error("Error reading uploaded CSV file.")
        st.stop()

# --------------------------------------------------
# Proceed only if dataset is loaded
# --------------------------------------------------
if "data" in st.session_state:
    data = st.session_state["data"]

    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Features", data.shape[1] - 1)
    c3.metric("Classes", data["target"].nunique())

    with st.expander("View Sample Data"):
        st.dataframe(data.head())

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # --------------------------------------------------
    # Train–Test Split
    # --------------------------------------------------
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=None
    )

    # --------------------------------------------------
    # Model Definitions
    # --------------------------------------------------
    if model_name == "Logistic Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced"
        )

    elif model_name == "KNN":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=9))
        ])

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced"
        )

    else:
        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )

    # --------------------------------------------------
    # Train & Predict
    # --------------------------------------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    st.subheader("📈 Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    c3.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
    c5.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    c6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.subheader("📄 Classification Report")
    st.dataframe(
        pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).transpose()
    )

else:
    st.info("⬆️ Download or upload a dataset to begin analysis.")
