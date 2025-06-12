import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, RocCurveDisplay
)

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def evaluate_models(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train and evaluate multiple classification models for Alzheimer's disease detection.

    This function:
    - Loads and preprocesses the dataset,
    - Splits the data into training and test sets,
    - Trains three models: Logistic Regression, Random Forest, and XGBoost,
    - Calculates evaluation metrics (accuracy, ROC AUC, precision, recall, F1 score),
    - Plots ROC curves for all models on a single figure,
    - Prints a summary comparison table.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with evaluation metrics for each model.

    Example
    -------
    >>> evaluate_models("data/alzheimers_disease_data.csv")

    Notes
    -----
    All models are trained and evaluated on the same train/test split for a fair comparison.
    ROC curves are plotted together for visual comparison.
    """
    # 1. Load data and apply feature engineering
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])  # feature matrix
    y = df['Diagnosis'].astype(int)                                    # target vector

    # 2. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # 3. Define models with chosen hyperparameters
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=2, random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=random_state
        )
    }

    # 4. Train and evaluate each model
    results = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, clf in models.items():
        # Each model is wrapped in a pipeline: preprocessing + classifier
        pipeline = Pipeline([
            ('preproc', build_preprocessing_pipeline()),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store results in a dictionary
        results.append({
            'model': name,
            'accuracy': acc,
            'roc_auc': auc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

        # Plot ROC curve for each model on the same axis
        RocCurveDisplay.from_estimator(
            pipeline, X_test, y_test,
            name=name, ax=ax
        )

    # Final plot adjustments and display
    ax.set_title("ROC Curves Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 5. Print and return summary table
    results_df = pd.DataFrame(results).set_index('model')
    print("\nModel comparison:\n", results_df)

    return results_df

if __name__ == "__main__":
    evaluate_models("../data/alzheimers_disease_data.csv")
