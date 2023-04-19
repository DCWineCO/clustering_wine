#Modules for data processing
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import acquire
import prepare
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


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
################################## Scale features ###########################  
def scale_and_concat(df, partitions):
    '''This functions uses the MinMaxScaler to scale the selected features and returns the new features
    scaled inside the dataframe'''
    # the variables that still need scaling
    scaled_vars = ['alcohol', 'volatileacidity', 'sulphates', 'density','chlorides','freesulfurdioxide']
     #select the X partitions: [X_train, X_validate, X_test]
    X= partitions[1:4] 
    # fit the minmaxscaler to X_train
    X_train = X[0]
    scaler = MinMaxScaler(copy=True).fit(X_train[scaled_vars])
    #create new column names for the scaled variables by adding 'scaled_' to the beginning of each variable name 
    scaled_column_names = ['scaled_' + i for i in scaled_vars]

    scaled_array = scaler.transform(df[scaled_vars])
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_column_names, index=df.index.values)
    return pd.concat((df, scaled_df), axis=1)

################################## Get Baseline RMSE ################################
def baseline(y_train, y_validate):
    '''This function creates your mean and median baselines '''
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 1. Predict quality_pred_mean
    quality_pred_mean = y_train['quality'].mean()
    y_train['quality_pred_mean'] = quality_pred_mean
    y_validate['quality_pred_mean'] = quality_pred_mean
    # 2. compute quality_pred_median
    quality_pred_median = y_train['quality'].median()
    y_train['quality_pred_median'] = quality_pred_median
    y_validate['quality_pred_median'] = quality_pred_median
    # 3. RMSE of quality_pred_mean
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", rmse_train, 
      "\nValidate/Out-of-Sample: ", rmse_validate)

    # 4. RMSE of quality_pred_median
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", rmse_train, 
      "\nValidate/Out-of-Sample: ", rmse_validate)
    
################################ Plot Actual vs Predictions #########################
def actualvs_pred(y_train):
    '''This function plots Mean and Median predictions vs actual '''
# plot to visualize actual vs predicted. 
    plt.hist(y_train.quality, color='blue', alpha=.5, label="Actual Quality",bins=range(1,10))
    plt.hist(y_train.quality_pred_mean, bins=1, color='red', alpha=.5, rwidth=100, label="Predicted Quality - Mean")
    plt.hist(y_train.quality_pred_median, bins=1, color='orange', alpha=.5, rwidth=100, label="Predicted Quality - Median")
    plt.xlabel("Qaulity (0-10)")
    plt.ylabel("Number of Wines")
    plt.legend()
    plt.show()
################################## Linear Regression Model ###########################
def lm_model(X_train,y_train,X_validate,y_validate):
    '''This function creates the Linear Regression Model'''
    # create the model object
    lm = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.quality)
    # predict train
    y_train['quality_pred_lm'] = lm.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lm)**(1/2)
    # predict validate
    y_validate['quality_pred_lm'] = lm.predict(X_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
############################# Lars Model ##################################
def lars_model(X_train,y_train,y_validate,X_validate):
    '''This function creates the Lasso Lars Model'''
    # create the model object
    lars = LassoLars(alpha=1.0)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.quality)
    # predict train
    y_train['quality_pred_lars'] = lars.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lars)**(1/2)
    # predict validate
    y_validate['quality_pred_lars'] = lars.predict(X_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
############################## Tweedie Model ##################################
def tweedie_model(X_train,y_train,y_validate,X_validate):
    '''
    This function creates the Tweedie Regressor Model
    '''
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.quality)
    # predict train
    y_train['quality_pred_glm'] = glm.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_glm)**(1/2)
    # predict validate
    y_validate['quality_pred_glm'] = glm.predict(X_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
###################################### Polynomial Model ##############################
def poly_model(X_train,y_train,y_validate,X_validate,X_test):
    '''This function creates the Polynomial Model'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.quality)
    # predict train
    y_train['quality_pred_lm2'] = lm2.predict(X_train_degree2)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lm2)**(1/2)
    # predict validate
    y_validate['quality_pred_lm2'] = lm2.predict(X_validate_degree2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
#################################### Ridge Model ##################################

def r_model(X_train,y_train,y_validate,X_validate):  
    '''This function creates the Ridge Model'''
    # create the model object
    R = Ridge()
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    R.fit(X_train, y_train.quality)
    # predict train
    y_train['quality_pred_R'] = R.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_R)**(1/2)
    # predict validate
    y_validate['quality_pred_R'] = R.predict(X_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_R)**(1/2)

    print("RMSE for R using Ridge \nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
####################################### SVR Model ###########################################
def svr_model(X_train,y_train,y_validate,X_validate):
    '''
    SVR Regression Model
    '''
    # create the model object
    svr = SVR()
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    svr.fit(X_train, y_train.quality)
    # predict train
    y_train['quality_pred_svr'] = svr.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_svr)**(1/2)
    # predict validate
    y_validate['quality_pred_svr'] = svr.predict(X_validate)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_svr)**(1/2)

    print("RMSE for svr using SVR \nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

####################################### Model Predictions vs Actual ############################
def plot_model_pred(y_validate):
    '''
    This function plots the actual vs model predictions 
    '''
    plt.figure(figsize=(16,9.5))
    plt.plot(y_validate.quality, y_validate.quality_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_validate.quality, y_validate.quality, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate.quality, y_validate.quality_pred_lars, 
            alpha=.5, color="red", s=100, label="Model: Lars")
    plt.scatter(y_validate.quality, y_validate.quality_pred_glm, 
            alpha=.5, color="yellow", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.quality, y_validate.quality_pred_lm, 
            alpha=.5, color="green", s=100, label="Linear Regression")
    plt.scatter(y_validate.quality, y_validate.quality_pred_lm2, 
            alpha=.5, color="purple", s=100, label="Polynomial")   
    
    plt.legend()
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Qualtiy")
    plt.title("Where are predictions more extreme? More modest?")
    plt.show()
########################################### Plot Error Change #########################
def plot_errors(y_validate):
    '''
    This function plots the error change as the actual value changes
    '''
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_validate.quality, y_validate.quality_pred_lm-y_validate.quality, 
            alpha=.5, color="red", s=100, label="Model: OLS Regression")
    plt.scatter(y_validate.quality, y_validate.quality_pred_glm-y_validate.quality, 
            alpha=.5, color="yellow", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.quality, y_validate.quality_pred_lm2-y_validate.quality, 
            alpha=.5, color="green", s=100, label="Model 2nd degree Polynomial")
    plt.scatter(y_validate.quality, y_validate.quality_pred_lars-y_validate.quality, 
            alpha=.5, color="blue", s=100, label="Model Lars")

    plt.legend()
    plt.xlabel("Actual Quality")
    plt.ylabel("Residual/Error: Predicted Quality - Actual Quality")
    plt.title("Do the size of errors change as the actual value changes?")
    plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()

###################################### Plot Top Models vs Actual ###############
def top_model(y_validate):
    '''This function plots the top two models vs the actual '''
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.quality, color='orange', alpha=.5, label="Actual Quality", bins=range(1,10))
    #plt.hist(y_validate.quality_pred_lars, color='red', alpha=.5, label="Model: Lars Regression", bins=range(1,10))
    plt.hist(y_validate.quality_pred_lm2, color='red', alpha=.5, label="Model: Polynomial", bins=range(1,10))
    #plt.hist(y_validate.quality_pred_lm, color='blue', alpha=.5, label="Model: OLS ", bins=range(1,10))
    plt.hist(y_validate.quality_pred_glm, color='pink', alpha=.5, label="Model: Tweedie", bins=range(1,10))
    #plt.hist(y_validate.quality_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial", bins=range(1,10)
    plt.xlabel("Quality")
    plt.ylabel("Number of Wines")
    plt.title("Comparing the Distribution of Actual Quality to Distributions of Predicted Quality for the Top Models")
    plt.legend()
    plt.show()
############################################## Best Model Predricions ##############################
def best_model(X_train,y_test,y_train,X_validate,X_test):
    '''This function comapare the best model to the out of sample test data. 
    Gives you model RMSE to be compared to baseline for results'''
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.quality)
    # predict train
    y_train['quality_pred_lm2'] = lm2.predict(X_train_degree2)


    # predict on test
    y_test['quality_pred_lm2'] = lm2.predict(X_test_degree2)
    y_test['quality_pred_lm2'] = y_test['quality_pred_lm2']
    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.quality, y_test.quality_pred_lm2)**(1/2)
    print("RMSE for OLS Model using Polynomial Regression\nOut-of-Sample Performance: ", rmse_test) 

######################### Percent of improvement of model on test data ###################
def final_model(y_test,y_validate):
    '''This function returns the percent change in rmse values for the model on test data'''
    rmse_test = mean_squared_error(y_test.quality, y_test.quality_pred_lm2)**(1/2)
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_mean)**(1/2)
    lm_base_ratio = (rmse_test/rmse_validate)*100
    print(f"""Polynomial Model decreased errors by {round(100-lm_base_ratio,2)}%""")

########################################### Pearson r ######################################
def pearson_r(train):
    '''This function runs a pearson correlation test on all features in relation to the target'''
    # Only compute pearson prod-moment correlations between feature
    # columns and target column
    target_col_name = 'quality'
    feature_target_corr = {}
    for col in train:
        if target_col_name != col:
            feature_target_corr[col + '_' + target_col_name] = \
                pearsonr(train[col], train[target_col_name])[0]
    print("Feature-Target Correlations")
    print(feature_target_corr)

########################################## Plot Feature to Target Relation ##########################
def relations_features(train):
    '''This function plots the relationships of features to the target'''
    # relation between each feature and the output
    feature_name = train.columns[:12]
    fig = plt.figure(figsize=(30,30))
    for i in range(1,12):
        axs = fig.add_subplot(4,4,i)
        sns.lineplot(x = train[feature_name[i-1]],y = train['quality'])

################################ Features to model on ###################################
def modeling_feats(X_train,X_validate,X_test):
    '''The function creates the variables for the features we will take into modeling.'''
    #Modeling features
    X_train = X_train[['fixedacidity', 'citricacid', 'residualsugar', 'totalsulfurdioxide',
       'pH', 'is_red', 'scaled_alcohol',
       'scaled_volatileacidity', 'scaled_sulphates', 'scaled_density',
       'scaled_chlorides', 'scaled_freesulfurdioxide']]

    X_validate = X_validate[['fixedacidity', 'citricacid', 'residualsugar', 'totalsulfurdioxide',
       'pH', 'is_red', 'scaled_alcohol',
       'scaled_volatileacidity', 'scaled_sulphates', 'scaled_density',
       'scaled_chlorides', 'scaled_freesulfurdioxide']]

    X_test = X_test[['fixedacidity', 'citricacid', 'residualsugar', 'totalsulfurdioxide',
       'pH','is_red', 'scaled_alcohol',
       'scaled_volatileacidity', 'scaled_sulphates', 'scaled_density',
       'scaled_chlorides', 'scaled_freesulfurdioxide']]