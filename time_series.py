import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def get_df(df, zipcode=None, city=None, state_initials=None, metro=None, county=None, start=None, end=None):
    """
    Input: 
    state_initials: string, 2 capital letters
    
    Returns:
    dataframe of zipcodes in the given state.
    """
    data = df
    
    if zipcode != None:
        data = data.loc[(df.RegionName == zipcode)]
    if city != None:
        data = data.loc[(df.City == city)]
    if state_initials != None:
        data = data.loc[(df.State == state_initials)]
    if metro != None: 
        data = data.loc[(df.Metro == metro)]
    if county != None:
        data = data.loc[(df.CountyName == county)]

    data.drop(columns=['RegionID', 'SizeRank'], inplace=True)
    #data.set_index('RegionName', inplace=True)
    head = data.iloc[:,:5]
    tail = data.iloc[:,5:]
    
    if start != None:
        i = tail.columns.get_loc(start)
    else:
        i = None
    if end != None:
        j = tail.columns.get_loc(end) + 1
    else:
        j = None
    tail = tail.iloc[:,i:j]
    
    new_df = pd.concat([head, tail], axis=1)
    new_df.set_index('RegionName', inplace=True)
    
    return new_df

def make_time_series(df):
    ts = pd.DataFrame()
    ts['Month'] = pd.to_datetime(df.columns.values[4:], format='%Y-%m')
    for zipcode in df.index:
        row = df[df.index==zipcode].iloc[:,4:]
        ts[zipcode] = row.values[0]
    ts.set_index('Month',inplace=True)
    return ts

def decompose_time_series(ts):
    decomposition = seasonal_decompose(ts)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    trend.plot(figsize=(20,4),legend=True,title='Trend')
    plt.title("Trend",{'size':22})
    plt.xlabel("Year",{'size':22})
    plt.legend(loc='lower left')
    plt.show()

    seasonal.plot(figsize=(20,4),legend=True,title='Seasonality')
    plt.title("Seasonality",{'size':22})
    plt.xlabel("Year",{'size':22})
    plt.legend(loc='lower left')
    plt.show()

    residual.plot(figsize=(20,4),legend=True,title='Residual')
    plt.title("Residual",{'size':22})
    plt.xlabel("Year",{'size':22})
    plt.legend(loc='lower left')
    plt.show()

def melt_data(df, start_index=0):
    """
    Input zipcode dataframe
    Converts from wide format to long format
    For use with time series modeling
    """
    melted = pd.melt(df, id_vars=['RegionID','RegionName', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], 
                     var_name='Time')
    melted['Time'] = pd.to_datetime(melted['Time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('Time').aggregate({'value':'mean'})[start_index:]

def plot_time_series(df, name=None, legend=None):
    # Instantiate a figure object. 
    plt.figure()
    if not legend:
        legend = list(df.columns)
    # Enumerate through each row in the dataframe passed in. Each row is a different zipcode.
    for ind, row in df.iterrows():
        # Get the median housing value data for the date ranges we want and store in a Series object
        data = pd.Series(row.iloc[7:])
        # Set the appropriate datetimes for data as the index for our data series
        data.index = pd.to_datetime(df.columns[7:])
        # Plot data for current zipcode on figure we instantiated on line 4. Set xticks to corresponding datetimes
        # Also make the figure large, so that we can read it more easily
        ax = data.plot(figsize=(20, 10), xticks=df.columns[7:])
        # add a label
        plt.ylabel("Median House Values")
        # let matplootlib autoformat the datetimes for the xticks
        plt.gcf().autofmt_xdate()
        # If name of city was provided, use it to set title of plot
        if name:
            plt.title("Median Home Value by Zip Code in {} from 1997-2013".format(name))
        else:
            plt.title("Median Home Value, 1997-2013")
    plt.legend(legend)        
    plt.show()


def stationarity_check(df, zipcode, threshold=.01):
    """
    p-value > 0.05: Fail to reject the null hypothesis, the data is non-stationary.
    p-value <= 0.05: Reject the null hypothesis, the data is stationary.
    """
    # Get time series
    TS = df[zipcode]

    # Calculate rolling statistics
    rolmean = TS.rolling(window = 8, center = False).mean()
    rolstd = TS.rolling(window = 8, center = False).std()
    
    # Perform the Dickey Fuller Test
    dftest = adfuller(TS)
    
    if dftest[1] < threshold:
        #Plot rolling statistics:
        fig = plt.figure(figsize=(12,6))
        orig = plt.plot(TS, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
        print('Reject the null hypothesis, the data is stationary.')
    else:
        print('Fail to reject the null hypothesis, the data is non-stationary.')
    
    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print('p-value = {}'.format(dfoutput[1]))
    
    return dftest[1]



def make_stationary(df, threshold=.01):
    d = 0
    for i in range(0,len(df)):
        print('d=', d)
        data_diff = df.diff(periods=i).dropna()
        dftest = stationarity_check(data_diff)
        if dftest < threshold:
            break
        else:
            d+=1
    print('d=', d)
    return d


#function to fit a SARIMAX model
def run_SARIMAX(ts,pdq,pdqs): 
    '''
    Fits a Statsmodels SARIMAX Time Series Model.
    
    Arguments:
         ts    = The time series being modeled.
         pdq   = list containing the p,d,and q values for the SARIMAX function.
         pdqs  = list containing the seasonal p,d and q values along with seasonal 
                 shift amount for the SARIMAX function.
    
    Return Value:
         Dictionary containing {model: model, terms: pdq, seasonal terms: pdqs, AIC: model AIC} values.
         Return AIC of 99999 and model=None indicates that the SARIMAX failed to execute.
    '''
    try:
        mod = sm.tsa.statespace.SARIMAX(ts,
                                       order=pdq,
                                       seasonal_order=pdqs,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
        output = mod.fit()
        return {'model':output,'terms':pdq,'seasonal terms':pdqs,'AIC':output.aic}
    
    except:
        return {'model':None,'terms':pdq,'seasonal terms':pdqs,'AIC':99999}

    
def get_SARIMA_parameters(ts,p,d,q,sp,sd,sq,s):
    '''
    Fits a Statsmodels SARIMAX Time Series Model for every combination of
    p,d,q and seasonal p,d,q values provided.
    
    Arguments:
         ts = The time series being modeled.
         p  = list of autoregressive terms to test
         d  = list of differencing parameter terms to test
         q  = list of moving average terms to test
         sp = list of seasonal autoregressive terms to test
         sd = list of seasonal differencing terms to test
         sq = list of seasonal moving average terms to test
         s  = list of seasonal durations to test
         
    Return Value:
         List of dictionaries containing (terms: pdq, seasonal terms: pdqs, AIC: model AIC) values.
         The list is sorted by lowest AIC to highest. 
         AIC of 99999 indicates that the SARIMAX failed to execute.
    '''
    
    #Make combinations of pdq and sp,sd,sq,s for the model
    combos = list(itertools.product(p, d, q))
    combos_seasonal = list(itertools.product(sp, sd, sq, s))
    print('total combos = ',len(combos) * len(combos_seasonal))
    
    #results will be appended to this empty list
    results = []
    
    #iterate through every combination, run a SARIMAX model and store the AIC value
    for combo in combos:
        for combo_season in combos_seasonal:
            result = run_SARIMAX(ts,combo,combo_season)
            #remove model because this function is only for parameter selection
            del result['model']
            results.append(result)
    
    #return the list sorted by AIC value
    return sorted(results,key=lambda x: x['AIC'])



def plot_prediction(ts,model,date,dynamic=False):
    '''
    Plot a prediction for the time series and model provided.
    
    Arguments:
         ts    = The time series being plotted.
         model = The model being used to make predictions.
         date  = The date after which predictions will be plotted
         dynamic = If True, use dynamic predictions from date provided
         
    Return Value:
         RMSE for the predicitons for the forecast
    
    '''
    pred = model.get_prediction(start=pd.to_datetime(date),dynamic=dynamic,full_results=True)
    pred_conf = pred.conf_int()
    ax = ts.plot(label='observed', figsize=(20,5))
    if dynamic:
        label='dynamic prediction from {}'.format(date)
    else:
        label='one-step-ahead prediction'
    pred.predicted_mean.plot(ax=ax, label=label,color='darkorange',alpha=0.9)
    ax.fill_between(pred_conf.index,pred_conf.iloc[:,0],pred_conf.iloc[:,1],color='b',alpha=0.1)
    plt.legend(loc='upper left')
    forecasted = pred.predicted_mean
    truth = ts[date :]
    RMSE = np.sqrt(np.mean((forecasted-truth)**2))
    print("RMSE: ",RMSE)
    return RMSE


def get_forecast(ts,model,steps,plot=False):
    '''
    Get an ROI 1-yr and 5-yr using a forcast with the time seriex and model provided.
    
    Arguments:
         ts    = The time series being forecasted.
         model = The model being used to make a forecast.
         steps = The number of steps ahead to forecast.
         plot  = If True, plot the forecast
         
    Return Value:
         Dictionary containing 1-yr and 5-yr ROI values.  
    '''
    prediction = model.get_forecast(steps=steps)
    pred_conf = prediction.conf_int()
    
    if plot:    
        #Plot future prediction with confidence interval
        ax = ts.plot(label='observed',figsize=(20,5))
        prediction.predicted_mean.plot(ax=ax,label='forecast')
        ax.fill_between(pred_conf.index,pred_conf.iloc[:,0],pred_conf.iloc[:,1],color='b',alpha=0.1)
        plt.legend(loc='upper left')
        plt.show()
    
    #At least 60 steps are required in order to calculate the 5-yr ROI
    if steps <= 60:
        prediction = model.get_forecast(steps=60)
    
    #Calculate 1yr and 5yr ROIs
    last_val = ts[-1]
    roi_1yr = round( (prediction.predicted_mean[11] - last_val)/last_val , 3)
    roi_5yr = round( (prediction.predicted_mean[59] - last_val)/last_val , 3)
    
    return {'ROI_1yr':roi_1yr, 'ROI_5yr': roi_5yr}


def plot_forecasts(df, model_dicts, steps, title=None):
    '''
    Plot forecasts for all zipcodes in results dictionary.
    
    Arguments:
         df         = The dataframe containing time series data with zipcodes as column names.
         model_dict = List containing dictionaries with models for each zipcode.
         steps      = The number of steps ahead to forecast.
         
    Return Value:
         A plot with all zipcodes in model_dicts  
    '''
    
    for item in model_dicts:
        zipcode = item['zipcode']
        label = f"{item['city']} {zipcode}"
        model = item['model']
        ts = df[zipcode]
        prediction = model.get_forecast(steps=steps)
        all_data = pd.concat([ts,prediction.predicted_mean],axis=0)
        ax = all_data.plot(label=label,figsize=(20,5))
    
    plt.axvline(x=df.index[-1], color='black', linestyle=':')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    return


def pacf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_pacf(ts, ax=ax, lags=lags)
    return


def acf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_acf(ts, ax=ax, lags=lags)
    return
