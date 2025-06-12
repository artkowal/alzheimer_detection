import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load Alzheimer's disease dataset from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.

    Example
    -------
    >>> df = load_data("data/alzheimers_disease_data.csv")
    """
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    df = load_data('../data/alzheimers_disease_data.csv')
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())

