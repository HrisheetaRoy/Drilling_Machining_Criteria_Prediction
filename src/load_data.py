import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def load_data() -> pd.DataFrame:
    df = pd.read_excel('../data/Drilling_data.xlsx')
    return df
