import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.feature_engineering import create_comorbidity_score, create_age_group

def preprocess_and_engineer(df):
    df = convert_categorical_dtype(df)
    df = create_comorbidity_score(df)
    df = create_age_group(df)
    return df

def convert_categorical_dtype(df: pd.DataFrame) -> pd.DataFrame:

    cat_cols = [
        'Gender', 'Ethnicity', 'EducationLevel',
        'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion',
        'Disorientation', 'PersonalityChanges',
        'DifficultyCompletingTasks', 'Forgetfulness', 'Diagnosis'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def build_preprocessing_pipeline() -> ColumnTransformer:
    ordinal_cols = ['EducationLevel']
    nominal_cols = [
        'Gender', 'Ethnicity', 'Smoking', 'FamilyHistoryAlzheimers',
        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
        'Hypertension', 'MemoryComplaints', 'BehavioralProblems',
        'Confusion', 'Disorientation', 'PersonalityChanges',
        'DifficultyCompletingTasks', 'Forgetfulness'
    ]
    numeric_cols = [
        'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
        'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
        'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
    ]


    ordinal_transformer = OrdinalEncoder()
    nominal_transformer = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols),
            ('num', scaler, numeric_cols)
        ],
        remainder='drop'
    )
    return preprocessor


if __name__ == "__main__":
    from src.data_import import load_data
    df = load_data("../data/alzheimers_disease_data.csv")
    df = convert_categorical_dtype(df)
    preproc = build_preprocessing_pipeline()
    X = df.drop(columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
    X_trans = preproc.fit_transform(X)
    print("Oryginal shape:", X.shape)
    print("Transformed shape:", X_trans.shape)
