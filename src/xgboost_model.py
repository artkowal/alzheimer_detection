import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)
from xgboost import XGBClassifier

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def train_xgboost(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train and evaluate an XGBoost model with hyperparameter optimization (GridSearchCV).

    Parameters
    ----------
    data_path : str
        Path to the CSV data file.
    test_size : float
        Proportion of the data to use as test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    grid : GridSearchCV object
        The fitted GridSearchCV containing the best XGBoost model.
    """
    # 1. Load data and perform feature engineering
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID','DoctorInCharge','Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3. Build the pipeline: preprocessing + XGBoost classifier
    pipeline = Pipeline([
        ('preproc', build_preprocessing_pipeline()),
        ('clf', XGBClassifier(eval_metric='logloss', random_state=random_state))
    ])

    # 4. Perform GridSearchCV for hyperparameter tuning
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 6, 10],
        'clf__learning_rate': [0.01, 0.1]
    }
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    # 5. Evaluation of the best model
    best = grid.best_estimator_
    print("Best XGB params:", grid.best_params_)

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1]

    print(f"XGB Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"XGB ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}\n")
    print("XGB Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nXGB Classification Report:\n", classification_report(y_test, y_pred))

    RocCurveDisplay.from_estimator(best, X_test, y_test)
    plt.title("ROC Curve – XGBoost")
    plt.show()

    return grid

if __name__ == "__main__":
    train_xgboost("../data/alzheimers_disease_data.csv")
