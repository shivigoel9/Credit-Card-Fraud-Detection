
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import kagglehub
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Automatically download dataset
print("\nDownloading Credit Card Fraud Dataset...")

try:
    existing_file = DATA_DIR / "creditcard.csv"

    if existing_file.exists():
        DATA_PATH = existing_file
        print("Using existing dataset!")

    else:
        download_path = kagglehub.dataset_download(
            "mlg-ulb/creditcardfraud"
        )

        csv_files = list(Path(download_path).rglob("creditcard.csv"))

        if not csv_files:
            raise FileNotFoundError("CSV file not found!")

        DATA_PATH = csv_files[0]

        # Copy dataset into project folder
        df = pd.read_csv(DATA_PATH)
        df.to_csv(existing_file, index=False)
        DATA_PATH = existing_file

    print("Dataset downloaded successfully!")

except Exception as e:
    print("\nDataset download failed!")
    print("Error:", e)
    print("Check your internet connection and Kaggle access.")
    sys.exit(1)

# Load dataset
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Visualization
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Genuine Transactions")
plt.savefig(OUTPUT_DIR / "class_distribution.png")
plt.close()

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []

# Training and evaluation
for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC-AUC:", roc_auc)

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = name.replace(" ", "_").lower()

    plt.savefig(
        OUTPUT_DIR / (filename + "_matrix.png")
    )
    plt.close()

    # Save model
    joblib.dump(
        model,
        OUTPUT_DIR / (filename + ".pkl")
    )

# Model comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("All files saved in outputs folder.")
