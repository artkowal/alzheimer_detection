import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.data_import import load_data
from src.preprocessing import build_preprocessing_pipeline, preprocess_and_engineer

def feature_importance_xgb(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    top_n: int = 20
):
    # 1. Load and engineer
    df = load_data(data_path)
    df = preprocess_and_engineer(df)
    X = df.drop(columns=['PatientID','DoctorInCharge','Diagnosis'])
    y = df['Diagnosis'].astype(int)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # 3. Build pipeline and fit
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

    # 4. Extract feature names
    feat_names = pipeline \
        .named_steps['preproc'] \
        .get_feature_names_out()

    # 5. Get importances
    importances = pipeline.named_steps['clf'].feature_importances_

    # 6. Build DataFrame and sort
    fi = pd.DataFrame({
        'feature': feat_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 7. Show top_n
    top = fi.head(top_n)
    print("\nTop features:\n", top)

    # 8. Plot
    plt.figure(figsize=(8, top_n * 0.3))
    plt.barh(top['feature'][::-1], top['importance'][::-1])
    plt.xlabel("Importance")
    plt.title("Top feature importances (XGBoost)")
    plt.tight_layout()
    plt.show()

    return fi

if __name__ == "__main__":
    feature_importance_xgb("../data/alzheimers_disease_data.csv")
