import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def shap_analysis(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Perform SHAP-based feature importance analysis using XGBoost.

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
    None
    """
    # 1. Load data and perform feature engineering
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID','DoctorInCharge','Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3. Build and train XGBoost pipeline
    pipeline = Pipeline([
        ('preproc', build_preprocessing_pipeline()),
        ('clf', XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=random_state
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 4. Preprocess the test set for SHAP (raw model input)
    X_test_trans = pipeline.named_steps['preproc'].transform(X_test)

    # 5. Initialize SHAP TreeExplainer with trained XGBoost model
    explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
    shap_values = explainer.shap_values(X_test_trans)

    # 6. Get feature names after preprocessing
    feature_names = pipeline.named_steps['preproc'].get_feature_names_out()

    # 7. Generate bar summary plot (global importance)
    shap.summary_plot(
        shap_values,
        X_test_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (bar)")
    plt.tight_layout()
    plt.show()

    # 8. Generate dot summary plot (impact & direction)
    shap.summary_plot(
        shap_values,
        X_test_trans,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Summary Plot (dot)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    shap_analysis("../data/alzheimers_disease_data.csv")
