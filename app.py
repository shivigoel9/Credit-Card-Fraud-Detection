
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

st.title("💳 Credit Card Fraud Detection System")
st.markdown("### AI-Powered Fraud Detection Dashboard")

st.divider()

# Check dataset
if not DATA_PATH.exists():
    st.error("Dataset not found! Run fraud_detection.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Dataset Explorer", "Model Performance", "Fraud Prediction"]
)

fraud_count = int((df["Class"] == 1).sum())
genuine_count = int((df["Class"] == 0).sum())
total = len(df)

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Transactions", f"{fraud_count:,}")
    col3.metric("Genuine Transactions", f"{genuine_count:,}")
    col4.metric("Fraud Percentage", f"{fraud_count/total*100:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Class", ax=ax)
        ax.set_xticks([0, 1], ["Genuine", "Fraud"])
        ax.set_xlabel("Transaction Type")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Fraud Percentage")
        fig, ax = plt.subplots()
        ax.pie(
            [genuine_count, fraud_count],
            labels=["Genuine", "Fraud"],
            autopct="%1.2f%%"
        )
        st.pyplot(fig)
        plt.close(fig)

    st.info("Fraud Detection Pipeline using SMOTE, Logistic Regression and Random Forest.")

elif page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    st.write("Dataset Shape:", df.shape)

    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Class Distribution")
    st.write(df["Class"].value_counts())

    st.download_button(
        "Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "creditcard.csv",
        "text/csv"
    )

elif page == "Model Performance":

    st.subheader("Model Performance Comparison")

    results_path = OUTPUT_DIR / "model_comparison.csv"

    if results_path.exists():

        results = pd.read_csv(results_path)
        st.dataframe(results, use_container_width=True)

        for metric in ["Precision", "Recall", "ROC-AUC"]:
            st.subheader(metric + " Comparison")

            fig, ax = plt.subplots()
            sns.barplot(
                data=results,
                x="Model",
                y=metric,
                ax=ax
            )
            ax.set_ylim(0, 1)
            plt.xticks(rotation=15)
            st.pyplot(fig)
            plt.close(fig)

        for model_name in [
            "logistic_regression",
            "random_forest"
        ]:

            image_path = OUTPUT_DIR / (model_name + "_matrix.png")

            if image_path.exists():
                st.subheader(model_name.replace("_", " ").title())
                st.image(str(image_path), use_container_width=True)

        st.download_button(
            "Download Model Comparison",
            results.to_csv(index=False).encode("utf-8"),
            "model_comparison.csv",
            "text/csv"
        )

    else:
        st.warning("Run fraud_detection.py first to generate results.")

elif page == "Fraud Prediction":

    st.subheader("Check a Transaction")

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        ["Logistic Regression", "Random Forest"]
    )

    model_file = (
        "logistic_regression.pkl"
        if model_choice == "Logistic Regression"
        else "random_forest.pkl"
    )

    model_path = OUTPUT_DIR / model_file

    if not model_path.exists():
        st.warning("Trained model not found. Run fraud_detection.py first.")

    else:

        model = joblib.load(model_path)

        st.write("Enter transaction feature values:")

        feature_columns = [c for c in df.columns if c != "Class"]

        with st.form("prediction_form"):

            values = {}

            columns = st.columns(3)

            for i, feature in enumerate(feature_columns):
                with columns[i % 3]:
                    values[feature] = st.number_input(
                        feature,
                        value=float(df[feature].median()),
                        format="%.6f"
                    )

            submitted = st.form_submit_button("Predict Transaction")

        if submitted:

            input_df = pd.DataFrame(
                [values],
                columns=feature_columns
            )

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.divider()

            if prediction == 1:
                st.error("🚨 FRAUDULENT TRANSACTION DETECTED!")
            else:
                st.success("✅ GENUINE TRANSACTION")

            st.metric(
                "Predicted Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.caption(
                "This is a machine learning prediction, not a confirmed financial determination."
            )
