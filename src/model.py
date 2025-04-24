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

    # 1. Wczytanie i feature engineering
    df = load_data(data_path)
    df = preprocess_and_engineer(df)  # konwersje, comorbidity_score, age_group

    # 2. Przygotowanie macierzy cech X i celu y
    X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 3. Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Pipeline: preprocessing + regresja logistyczna
    preprocessor = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ))
    ])

    # 5. Trening modelu
    pipeline.fit(X_train, y_train)

    # 6. Predykcje
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 7. Ocena – metryki
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:            {acc:.4f}")
    print(f"ROC AUC:             {auc:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Wykres ROC
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve – Logistic Regression")
    plt.show()

    return pipeline

if __name__ == "__main__":
    # Uruchomienie skryptu:
    # python -m src.model
    train_baseline_model("../data/alzheimers_disease_data.csv")
