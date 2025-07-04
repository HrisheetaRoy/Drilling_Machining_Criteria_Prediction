import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Strip whitespace and lowercase column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Handle missing values (if any)
    df = df.dropna() 
    
    # Remove outliers in torque_(nm) using Z-score
    if 'torque_(nm)' in df.columns:
        z_scores = np.abs(stats.zscore(df['torque_(nm)']))
        df = df[z_scores < 3]  # Keep rows with Z < 3

    # Remove flat/noisy flank wear data (e.g., below measurement threshold)
    if 'flank_wear_(mm)' in df.columns:
        df = df[df['flank_wear_(mm)'] > 0.01]  # Remove values ≤ 0.01 mm

    return df
