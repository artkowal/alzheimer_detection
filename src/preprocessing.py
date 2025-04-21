import pandas as pd

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


if __name__ == "__main__":
    from src.data_import import load_data
    df = load_data("../data/alzheimers_disease_data.csv")
    df = convert_categorical_dtype(df)
    print(df.dtypes[df.dtypes == 'category'])
