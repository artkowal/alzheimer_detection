import pandas as pd

def create_comorbidity_score(df: pd.DataFrame) -> pd.DataFrame:

    disease_cols = [
        'Hypertension',
        'Diabetes',
        'CardiovascularDisease',
        'Depression',
        'HeadInjury',
        'FamilyHistoryAlzheimers'
    ]
    df['comorbidity_score'] = df[disease_cols].sum(axis=1)
    return df

def create_age_group(df: pd.DataFrame, bins=None, labels=None) -> pd.DataFrame:
    if bins is None:
        bins = [60, 70, 80, 90]
    if labels is None:
        labels = ['60-69', '70-79', '80-90']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

if __name__ == '__main__':

    from src.data_import import load_data
    df = load_data("../data/alzheimers_disease_data.csv")
    df = create_comorbidity_score(df)
    df = create_age_group(df)
    print(df[['Hypertension', 'Diabetes', 'CardiovascularDisease',
              'Depression', 'HeadInjury', 'FamilyHistoryAlzheimers',
              'comorbidity_score', 'Age', 'age_group']].head())
