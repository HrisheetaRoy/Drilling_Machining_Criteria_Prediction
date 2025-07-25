# Winsorizing using Interquartile Range-based thresholds [especially good method when the data has skewed distributions or natural outliers]

import pandas as pd

def cap_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25) #calc 1st  quartile of the specified col
    Q3 = df[column].quantile(0.75) #calc 3rd  quartile of the specified col
    IQR = Q3 - Q1 #InterQuartileRange
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    print(f"[Capping] {column} → Q1: {Q1:.2f}, Q3: {Q3:.2f}, Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# This line caps values:

# Anything below Q1 - 1.5×IQR is replaced with the lower bound

# Anything above Q3 + 1.5×IQR is replaced with the upper bound