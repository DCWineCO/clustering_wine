<<<<<<< HEAD
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##################### Prepare Wine Data ##################################
def outlier_function(df, cols, k):
    '''
    This function removes white space in column names, takes in a dataframe, column, and k
    to detect and handle outlier using IQR rule
    '''
    df.columns = df.columns.str.replace(' ', '')
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]

def prepare(df, target_var):
    '''
    This function removes white space in column names , checks and removes outliers and 
    takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''

    df.columns = df.columns.str.replace(' ', '')
    cols=['fixedacidity',
                     'volatileacidity',
                     'citricacid',
                     'residualsugar',
                     'chlorides',
                     'freesulfurdioxide',
                     'totalsulfurdioxide',
                     'density',
                     'pH',
                     'sulphates',
                     'alcohol',
                     'quality']
    k=1.5
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]

    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions

##############################################################

=======
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
    
>>>>>>> 797893a79feb041a5a2c152e2d76779870ae9d7d
