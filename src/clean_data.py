import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.dropna()

    if 'workpiece' in df.columns:
        df = pd.get_dummies(df, columns=['workpiece'], drop_first=True)

    if 'torque_(nm)' in df.columns:
        z_scores = np.abs(stats.zscore(df['torque_(nm)']))
        df = df[z_scores < 3]

    if 'flank_wear_(mm)' in df.columns:
        df = df[df['flank_wear_(mm)'] > 0.01]

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['speed_per_dia'] = df['speed_(rpm)'] / df['diameter_(mm)']
    df['log_feed'] = np.log(df['feed_(mm/rev)'] + 1e-5)

    if 'torque_(nm)' in df.columns and 'feed_(mm/rev)' in df.columns:
        df['torque_feed_ratio'] = df['torque_(nm)'] / (df['feed_(mm/rev)'] + 1e-5)

    if 'thrust_force_(n)' in df.columns and 'torque_(nm)' in df.columns:
        df['material_hardness'] = df['thrust_force_(n)'] / (df['torque_(nm)'] + 1e-5)

    df['feed_squared'] = df['feed_(mm/rev)'] ** 2
    df['speed_squared'] = df['speed_(rpm)'] ** 2

    return df
