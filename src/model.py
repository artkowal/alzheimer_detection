import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def train_baseline_model(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Pipeline:
    """
    Train and evaluate a baseline logistic regression model for Alzheimer’s detection.

    Parameters
    ----------
    data_path : str
        Path to the CSV file with data.
    test_size : float, optional
        Fraction of data reserved for test split. Default is 0.2.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    Pipeline
        Trained pipeline (preprocessing + logistic regression).
    """
    # 1. Load and preprocess data
    df = load_data(data_path)
    df = preprocess_and_engineer(df)  # convert categories, add features

    # 2. Separate features and target
    X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Pipeline: preprocessing + logistic regression
    preprocessor = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ))
    ])

    # 5. Train model
    pipeline.fit(X_train, y_train)

    # 6. Predict and calculate probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 7. Print evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:            {acc:.4f}")
    print(f"ROC AUC:             {auc:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Plot ROC curve
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve – Logistic Regression")
    plt.show()

    return pipeline

if __name__ == "__main__":
    # Run the script directly to train and evaluate the baseline model.
    train_baseline_model("../data/alzheimers_disease_data.csv")
