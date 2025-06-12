import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def train_random_forest(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train and evaluate a RandomForestClassifier with hyperparameter tuning via GridSearchCV.

    Parameters
    ----------
    data_path : str
        Path to the CSV data file.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object for RandomForest.
    """
    # 1. Load and feature engineer the data
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID','DoctorInCharge','Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # 3. Build a pipeline: preprocessing + random forest classifier
    pipeline = Pipeline([
        ('preproc', build_preprocessing_pipeline()),
        ('clf', RandomForestClassifier(random_state=random_state))
    ])

    # 4. Grid search for hyperparameters
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    # 5. Evaluation of the best model
    best = grid.best_estimator_
    print("Best RF params:", grid.best_params_)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1]

    print(f"RF Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"RF ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}\n")
    print("RF Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nRF Classification Report:\n", classification_report(y_test, y_pred))

    # ROC curve for visualization
    RocCurveDisplay.from_estimator(best, X_test, y_test)
    plt.title("ROC Curve – Random Forest")
    plt.show()

    return grid

if __name__ == "__main__":
    train_random_forest("../data/alzheimers_disease_data.csv")
