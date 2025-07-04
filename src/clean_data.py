import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Strip whitespace and lowercase column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Handle missing values (if any)
    df = df.dropna() 

    return df
