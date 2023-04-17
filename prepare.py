# prepare.py
import numpy as np
import pandas as pd

def fix_col_names(df):
    cols = df.columns.str.replace(' ', '_')
    cols = dict(zip(df.columns, cols))
    df.rename(columns=cols, inplace=True)
    
    return df


def prep_wine_data(df):
    df = fix_col_names(df)
    
    return df
    