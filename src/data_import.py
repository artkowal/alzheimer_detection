import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load Alzheimer’s dataset from a CSV file.

    Parameters:
    -----------
    path : str
        Path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    df = load_data('../data/alzheimers_disease_data.csv')
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())

