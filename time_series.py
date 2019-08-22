# Imports

import pandas as pd
import numpy as np
import random
random.seed(42)

#Suppress scientific notation
pd.options.display.float_format = '{:.3f}'.format

# Visualizations
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn; seaborn.set()
from statsmodels.tsa.seasonal import seasonal_decompose

# Modeling
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.tsa.api as smt
import scipy.stats as scs

# Progress bars
import time
from tqdm import tqdm

# Remove warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
    
    
# Functions    
    
# Customize dataset in one line
def get_df(df, zipcode=None, city=None, state_initials=None, metro=None, county=None, start=None, end=None):
    '''
    Input: 
    df: dataframe with format from 'zillow_data.csv' 
    zipcode: int, 5 digits
    city: string
    state_initials: string, 2 capital letters
    metro: string
    county: string
    start: string, date format 'YYYY-MM'
    end: string, date format 'YYYY-MM'
    
    Returns:
    Dataframe of given location parameters and start/end dates
    '''
    data = df.copy()
    #Find specified location
    if zipcode:
        data = data.loc[(df.RegionName == zipcode)]
    if city:
        data = data.loc[(df.City == city)]
    if state_initials:
        data = data.loc[(df.State == state_initials)]
    if metro: 
        data = data.loc[(df.Metro == metro)]
    if county:
        data = data.loc[(df.CountyName == county)]
    #Drop unnecessary columns
    data.drop(columns=['RegionID', 'SizeRank'], inplace=True)
    #Split dataframe into location information and house values
    head = data.iloc[:,:5]
    tail = data.iloc[:,5:]
    #Limit time range to given start/end dates
    if start:
        #Get index of start date
        i = tail.columns.get_loc(start)
    else:
        i = None
    if end:
        #Get index of end date
        j = tail.columns.get_loc(end) + 1
    else:
        j = None
    #Slice tail with given start/end indexes
    tail = tail.iloc[:,i:j]
    #Combine head and tail
    new_df = pd.concat([head, tail], axis=1)
    #Set zipcode as index
    new_df.set_index('RegionName', inplace=True)
    return new_df

# Convert dataframe to a usable format for time series analysis
def make_time_series(df):
    '''
    Input:
    Dataframe with format from 'zillow_data.csv' 
    
    Return:
    Time series format with 
    index set as Month and 
    columns for each zipcode in df
    '''
    ts = pd.DataFrame()
    #Set column Month to dates from column names in df, convert to datetime
    ts['Month'] = pd.to_datetime(df.columns.values[4:], format='%Y-%m')
    for zipcode in df.index:
        #For each zipcode, find all the housing values
        row = df[df.index==zipcode].iloc[:,4:]
        #Make column for each zipcode with housing values
        ts[zipcode] = row.values[0]
    ts.set_index('Month',inplace=True)
    return ts

# Plot timeseries of house value by zipcode
def plot_time_series(ts, region_name=None, line=True, boxplot=True, figsize=(12,8)):
    '''
    Plot line graph and boxplot of time series region
    
    Input:
    ts: time series format
    region_name: string, name of target region
    line: boolean, plot line graph
    boxplot: boolean, plot boxplot
    figsize: default (12,8)
    '''
    if line:
        # Generate line graph for each zipcode
        ts.plot(figsize=figsize)
        if region_name:
            plt.title("Median Home Value by Zip Code in {}".format(region_name))
        else:
            plt.title("Median Home Value")
        plt.show()

    if boxplot:
        # Generate a box and whiskers plot for each zipcode
        ts.boxplot(figsize = figsize)
        if region_name:
            plt.title("Median Home Value by Zip Code in {}".format(region_name))
        else:
            plt.title("Median Home Value")
        plt.show()

def decompose_time_series(ts, figsize=(12,4), mean=False):
    if mean==True:
        decomposition = seasonal_decompose(ts.mean(axis=1))
    else:
        decomposition = seasonal_decompose(ts)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    trend.plot(figsize=figsize)
    plt.title("Trend")
    plt.xlabel("Year")
    plt.show()

    seasonal.plot(figsize=figsize)
    plt.title("Seasonality")
    plt.xlabel("Year")
    plt.show()

    residual.plot(figsize=figsize)
    plt.title("Residual")
    plt.xlabel("Year")
    plt.show()

def pacf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_pacf(ts, ax=ax, lags=lags)
    return


def acf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_acf(ts, ax=ax, lags=lags)
    return

def train_test_split(ts, len_test):
    train, test = ts[:-len_test], ts[-len_test:]
    print("Train Test Split Complete \nLength of Train: {} \tLength of Test: {}".format(len(train), len(test)))
    return train, test

# create a list of parameters to try
def param_combinations(param_range=3, differencing_range=2, seasonal=[0,12], trend=[None,'t','ct']):
    '''
    Creates combinations of parameters for SARIMAX modeling
    
    Input: 
        param_range: int, range for p, q, P, Q (default 3=[0,1,2])
        differencing_range: int, range for d and D (default 2=[0,1])
        seasonal: list, default [0,12]
        trend: list, default [None,'t','ct']
            None - SARIMAX default
            't' - linear trend
            'ct' - linear trend with constant
            *Note: to use only None, enter [None]
        
    Return:
        list in this format: [(p,d,q), (P,D,Q,s), t]
    '''
    p = q = P = Q = range(param_range) #default 3 [0,1,2]
    d = D = range(differencing_range) #default 2 [0,1]
    s = seasonal #default [0,12]
    t = trend #default [None,'t','ct']
    params = []
    # create config instances
    for p_ in tqdm(p, desc='Making parameter combinations', leave=False):
        for d_ in d:
            for q_ in q:
                for t_ in t:
                    for P_ in P:
                        for D_ in D:
                            for Q_ in Q:
                                for s_ in s:
                                    combo = [(p_,d_,q_), (P_,D_,Q_,s_), t_]
                                    params.append(combo)
    return params

# root mean squared error or rmse
def measure_rmse(true_values, predictions):
    return np.sqrt(mean_squared_error(true_values, predictions))

def sarimax(ts, order, sorder, trend=None):
    '''
    Fits a Statsmodels SARIMAX Time Series Model.
    
    Inputs:
        ts: time series format
        order: list containing the p,d,and q values for the SARIMAX function.
        sorder: list containing the seasonal p,d and q values along with seasonal 
            shift amount for the SARIMAX function
        trend: string, options=[None, 'n', 'c', 'ct', 't']
    
    Return:
        fitted model
    '''
    #Run and fit SARIMAX model with given parameters
    model = sm.tsa.statespace.SARIMAX(ts, order=order, seasonal_order=sorder, trend=trend, 
                                      enforce_stationarity=False, 
                                      enforce_invertibility=False)
    fit_model = model.fit(disp=False)
    return fit_model

def cross_validation_rmse(ts, order, sorder, trend=None, n_splits=4):
    '''
    Uses time series cross validation (TimeSeriesSplit) of n_splits.
    Calculates RMSE for each split
    *Remember to reserve a portion of the data for final model evaluation
    
    Input:
        ts - full time series
        order - ints, (#,#,#)
        sorder - ints, (#,#,#,#)
        trend - string
        n_splits - number of cross validation splits (default 4)
        
    Returns:
        Average root mean squared error of cross validations 
    '''
    #Initialize TimeSeriesSplit with n_splits (default 4)
    tscv = TimeSeriesSplit(n_splits = n_splits)
    #Create list for RMSE of each cross validation split
    rmse = []
    try:
        #Use tscv to implement Forward Chaining Nested Cross Validation
        for train_index, test_index in tscv.split(ts):
            #Make train, test split for section
            cv_train, cv_test = ts.iloc[train_index], ts.iloc[test_index]
            #Run and fit model for section
            model = sarimax(cv_train, order, sorder, trend)
            #Get predictions from model for cv_test date range
            predictions = model.predict(cv_test.index.values[0], cv_test.index.values[-1])
            #Store true values for cv_test date range
            true_values = cv_test.values
            #Calculate RMSE and append to list
            rmse.append(measure_rmse(true_values, predictions))
        #Calculate average of RMSEs from cross validations
        cv_rmse = round(sum(rmse)/n_splits,3)
    except:
        cv_rmse = np.nan
    
    return cv_rmse

def results_dict(ts, order, sorder, trend):
    cv_rmse = cross_validation_rmse(ts, order, sorder, trend)
    model = sarimax(ts, order, sorder, trend)
    aic = model.aic
    bic = model.bic
    dictionary = {'model':model,'order':order,'sorder':sorder,'trend':trend,
                'AIC':aic,'BIC':bic, 'CVRMSE':cv_rmse}
    return dictionary

def results_dict(ts_all, order, sorder, trend):
    train_test_split_test_split()
    cv_rmse = cross_validation_rmse(ts, order, sorder, trend)
    model = sarimax(ts, order, sorder, trend)
    aic = model.aic
    bic = model.bic
    dictionary = {'model':model,'order':order,'sorder':sorder,'trend':trend,
                'AIC':aic,'BIC':bic, 'CVRMSE':cv_rmse}
    return dictionary

def run_models_by_params(ts, param_combos):
    '''
    Function to run SARIMAX model with cross validation for all parameter combinations
    for single time series
    Input:
        ts: single time series (training data)
        param_combos: list of parameter combinations
            format: order, sorder, trend
            
    '''
    #Initialize list for results
    results = []
    #Iterate through parameters with progress baar
    for param in tqdm(param_combos, desc='Running models with cross validation', leave=False):
        #Separate parameters
        order, sorder, trend = param
        #Run model with cross validation
        result = results_dict(ts, order, sorder, trend)
        #Add dictionary of result to list
        results.append(result)
    #Convert list of dictionaries to dataframe
    df = pd.DataFrame(results, columns=['model', 'order', 'sorder', 'trend', 'AIC', 'BIC', 'CVRMSE'])
    return df

def run_all_models(ts_all, param_combos):
    '''
    Function to iterate through zipcodes and run SARIMAX models with cross validation 
    for all combinations of parameters
    
    Input:
        ts - time series of region (training data)
        param_combos - list of parameter combinations
    '''
    #Initialize list for rows
    df_list = []
    #Iterate through the columns in the time seres
    for zipcode in tqdm(ts_all.columns, desc='Modeling zipcodes', leave=False):
        #Sanity check
        print(f'Running models for zipcode {zipcode}')
        #Isolate the data for the zipcode
        ts = ts_all[zipcode]
        #Iterate through all param_combos using time series cross validation
        #Stores row as dataframe with results of model
        zip_df = run_models_by_params(ts, param_combos)
        #Add column for zipcode to dataframe
        zip_df.insert(0, 'zipcode', zipcode)
        #Add row to df_list
        df_list.append(zip_df)
    #Combine zip_df into dataframe
    df = pd.concat(df_list)
    return df

def sort_best_models(results_df, criterion, drop_duplicates=True):
    '''
    Input:
        df - dataframe of all model results
        criterion - string, column name to sort by
        
    Returns:
        Dataframe of best model results for each zipcode
    '''
    df = results_df.copy()
    #Drop nan values in CVRMSE columns
    df.dropna(subset=['CVRMSE'], inplace=True)
    #Sort values by given criterion
    df.sort_values(criterion, ascending=True, inplace=True)
    if drop_duplicates:
        #Get top row for each zipcode
        df.drop_duplicates(['zipcode'], inplace=True)
    #Fill null values with None (affects trend)
    df.fillna('None', inplace=True)
    return df

def extract_params(results_df, zipcode):
    '''
    Input:
        best_results - Dataframe from running models
        zipcode - int, 5 digits
        
    Returns:
        order, seasonal_order, trend
    '''
    row = results_df.loc[results_df['zipcode']==zipcode]
    order = (int(row['order'].values[0][1]),
            int(row['order'].values[0][4]),
            int(row['order'].values[0][7]))
    sorder = (int(row['sorder'].values[0][1]),
            int(row['sorder'].values[0][4]),
            int(row['sorder'].values[0][7]),
            int(row['sorder'].values[0][10:12]))
    trend = str(row['trend'].values[0])    
    #print(f'zipcode: {zipcode}, order: {order}, sorder: {sorder}, trend: {trend}')
    
    return order,sorder,trend

def rmse_final(train, test, order, sorder, trend=None):
    model = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=sorder, trend=trend, 
                                      enforce_stationarity=False, 
                                      enforce_invertibility=False)
    fit_model = model.fit(disp=False)
    #Get predictions from model for test date range
    predictions = fit_model.predict(test.index.values[0], test.index.values[-1])
    #Store true values for test date range
    true_values = test.values
    #Calculate RMSE
    rmse = measure_rmse(true_values, predictions)
    return rmse

def add_rmse_to_final_models(best_results, train, test):
    #Add final RMSE for each model
    rmse_list = []
    best_results.dropna(subset=['CVRMSE'], inplace=True)
    for zipcode in best_results.zipcode:
        o,s,t = extract_params(best_results, zipcode)
        rmse = rmse_final(train[zipcode], test[zipcode], o, s, None)
        rmse_list.append(rmse)
    best_results['RMSE'] = np.round(rmse_list,1)

def plot_predictions(best_results_cvrmse, ts_all, start_test='2017-04', 
                     start_pred='2017-06-01', end_pred=None, dynamic=False):
    fig, axes = plt.subplots(nrows=9, ncols=2, figsize=(15,65))
    axes_list = [item for sublist in axes for item in sublist] 
    train, test = train_test_split(ts_all, 12)
    for zipcode in best_results_cvrmse.zipcode:
        ax = axes_list.pop(0)
        order,sorder,trend = extract_params(best_results_cvrmse, zipcode)
        output = sarimax(ts_all[zipcode], order, sorder)
        pred = output.get_prediction(start=start_pred, end=end_pred, dynamic=dynamic)
        pred_ci = pred.conf_int()
        test[zipcode][start_test:].plot(label='Observed', ax=ax)
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], 
                        color='k', alpha=.2)

        ax.set_ylabel(f'Housing Value for {zipcode}')
        ax.set_title(f'Model Validation for Zipcode {zipcode}')


    # Now use the matplotlib .remove() method to 
    # delete anything we didn't use
    for ax in axes_list:
        ax.remove()

def plot_roi(best_results_cvrmse, ts_all):
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15,36))
    axes_list = [item for sublist in axes for item in sublist] 

    for zipcode in best_results_cvrmse.zipcode:
        ax = axes_list.pop(0)
        ROI_1yr = 100 * (ts_all[zipcode].tshift(-12) / ts_all[zipcode] - 1)
        ROI_1yr.plot(x='Year', y='ROI', label='1 Year ROI', ax=ax, legend=True)
        ROI_2yr = 100 * (ts_all[zipcode].tshift(-24) / ts_all[zipcode] - 1)
        ROI_2yr.plot(x='Year', y='ROI', label='2 Year ROI', ax=ax, legend=True)
        ax.set_title(zipcode)
        ax.axhline(c='black')
        ax.tick_params(
            which='both',
            bottom='on',
            left='on',
            right='off',
            top='off'
        )
        ax.set_ylim(-30, 70)
        ax.set_ylabel('Return on Investment')
        #ax.spines['left'].set_position('zero')

        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)


    # Now use the matplotlib .remove() method to 
    # delete anything we didn't use
    for ax in axes_list:
        ax.remove()