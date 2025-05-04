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
    # 1. Load & engineer
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID','DoctorInCharge','Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3. Train final XGB
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

    # 4. Preprocess test set to raw model input
    X_test_trans = pipeline.named_steps['preproc'].transform(X_test)

    # 5. SHAP explainer
    explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
    shap_values = explainer.shap_values(X_test_trans)

    # 6. Feature names
    feature_names = pipeline.named_steps['preproc'].get_feature_names_out()

    # 7. Summary plot
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

    # 8. Dot summary (kierunek wpływu)
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
