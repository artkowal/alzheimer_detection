import pandas as pd

def create_comorbidity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comorbidity score for each patient by summing the presence
    of several chronic diseases or risk factors (binary columns).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for chronic diseases and risk factors.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'comorbidity_score' which
        indicates the total number of comorbidities for each patient.

    Notes
    -----
    The following binary columns are included in the comorbidity score:
    - 'Hypertension'
    - 'Diabetes'
    - 'CardiovascularDisease'
    - 'Depression'
    - 'HeadInjury'
    - 'FamilyHistoryAlzheimers'
    """
    disease_cols = [
        'Hypertension',
        'Diabetes',
        'CardiovascularDisease',
        'Depression',
        'HeadInjury',
        'FamilyHistoryAlzheimers'
    ]
    # Convert all relevant columns to integer (in case they're categorical)
    # and sum them row-wise to get the total number of comorbidities.
    df['comorbidity_score'] = df[disease_cols].apply(lambda col: col.astype(int)).sum(axis=1)
    return df

def create_age_group(df: pd.DataFrame, bins=None, labels=None) -> pd.DataFrame:
    """
    Add an 'age_group' column to the DataFrame by binning the 'Age' variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column 'Age'.
    bins : list or None, optional
        List of bin edges for age groups. Defaults to [60, 70, 80, 90].
    labels : list or None, optional
        List of labels for each age group. Defaults to ['60-69', '70-79', '80-90'].

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'age_group' column indicating
        which age group each patient belongs to.

    Notes
    -----
    The pd.cut() function is used for binning ages into specified intervals.
    """
    if bins is None:
        bins = [60, 70, 80, 90]
    if labels is None:
        labels = ['60-69', '70-79', '80-90']
    # pd.cut divides the 'Age' values into discrete bins with labels.
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

if __name__ == '__main__':
    # Example usage / test block: run this file directly to see example output.
    from src.data_import import load_data
    df = load_data("../data/alzheimers_disease_data.csv")
    df = create_comorbidity_score(df)
    df = create_age_group(df)
    # Print the first five rows showing comorbidities, age, and age group.
    print(df[['Hypertension', 'Diabetes', 'CardiovascularDisease',
              'Depression', 'HeadInjury', 'FamilyHistoryAlzheimers',
              'comorbidity_score', 'Age', 'age_group']].head())
