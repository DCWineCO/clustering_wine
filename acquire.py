# acquire.py

import numpy as np
import pandas as pd
import os

def new_wine_data():
    '''
    This function queries data from two data.world datasets and reads
    them into pandas DataFrames. One is for red wine 
    (https://query.data.world/s/hn4uhqtyxklbrtbdbxmku47vueahfw?dws=00000)
    The other is for white wine.
    (https://query.data.world/s/r2mrliiyey6g2rn54wqmx3pvcylez7?dws=00000)
    The function then adds a column 'red_wine' to each df to designate 
    whether each wine is white or red. Then the two dfs are concatenated
    into a single dataframe which is returned
    
    Arguments: None
    
    Returns: DataFrame of properties queried
    '''
    
    # Read in red wine data from data.world.
    df_r = pd.read_csv(
        'https://query.data.world/s/572bfogx33kophnqyp3lwse7pguchi?dws=00000')
    
    # Read in white wine data from data.world.
    df_w = pd.read_csv(
        'https://query.data.world/s/r2mrliiyey6g2rn54wqmx3pvcylez7?dws=00000')
    
    # Add 'is_red' column to each df'
    df_r['is_red'] = 1
    df_w['is_red'] = 0
    
    # Concatenate two dfs
    df = pd.concat([df_r, df_w])
    
    return df


def get_wine_data():
    '''
    This function checks to see if there is a local version of 'wine.csv'.
    If it finds one, it reads it into a DataFrame and returns that df.
    If it does not find one, it runs 'new_wine_data()' to pull the data
    from the host and convert to a df. Then it writes that df to a local
    file 'wine.csv' and returns the df. Function relies
    on other functions in the wrangle.py module.
    '''
    if os.path.isfile('wine.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('wine.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_wine_data()
        
        # Cache data
        df.to_csv('wine.csv')
        
    return df