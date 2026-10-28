from data import build_final_df, get_transformed_series, select_daily_demand, build_master_data, get_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import data
from data import plot_acf, plot_pacf

from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from datetime import timedelta
from datetime import datetime

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ORDER = (1, 1 ,1)
SEASONAL_ORDER = (1, 0, 7, 7)
OUTPUT_DIR = Path(__file__).resolve().parent / "Outputs"
TRAIN_END = datetime(2026, 5, 23)
TEST_END = datetime(2026, 8, 31)
EXOG_VARIABLES = ["temperature_2m_mean", "temperature_2m_max", "precipitation_sum", "snowfall_sum", "wind_speed_10m_max", "holiday_day"]						

def initialise_data():
    global daily_rides, daily_rides_transformed, pt, daily_rides_transformed_diffed
    global TRAIN, TEST, ENDOG, EXOG

    daily_rides = select_daily_demand()
    daily_rides_transformed, pt = get_transformed_series()
    daily_rides_transformed_diffed = (daily_rides_transformed).diff()[1:]
    master_data = build_master_data(daily_rides_transformed)

    TRAIN = daily_rides_transformed[:TRAIN_END]
    TEST = daily_rides[TRAIN_END + timedelta(days =  1):TEST_END]
    ENDOG = daily_rides_transformed
    EXOG = master_data[EXOG_VARIABLES]

    ENDOG.index = pd.to_datetime(ENDOG.index)
    EXOG.index = pd.to_datetime(EXOG.index)
    TEST.index = pd.to_datetime(TEST.index)


def eda_plots():
    fig, ax = plt.subplots(figsize = (16,8))
    ax.plot(daily_rides)
    ax.set_title('NYC Daily Yellow Taxi Demand (2024-10-1 to 2026-08-31)', fontsize = 16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    fig.savefig(OUTPUT_DIR/"Daily Rides Demand Series.png", dpi = 300)
    plt.tight_layout()
    plt.close(fig)

    fig, ax = plt.subplots(2, figsize = (17,12), constrained_layout = True)
    ax[0].plot(daily_rides_transformed)
    ax[0].set_title('Normalised Daily Demand', fontsize = 16)
    ax[1].plot(pd.Series(daily_rides_transformed).diff()[1:])
    ax[1].set_title('Normalised Daily Demand with Single Period Differencing', fontsize = 16)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR/"Normalised and Differenced Series.png", dpi = 300)

def correlelogram_plots():
    fig,ax = plt.subplots(2, figsize = (12,8))
    plot_pacf((daily_rides_transformed_diffed), ax = ax[0]);
    plot_acf((daily_rides_transformed_diffed), ax = ax[1]);
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR/"ACF and PACF Plots.png", dpi = 300)

def decomposition_plot(): 
    decomposition = STL(daily_rides, period = 7).fit() 
    fig, ax = plt.subplots(4, figsize = (16,12), constrained_layout = True)
    ax[0].plot(decomposition.observed)
    ax[0].set_title("Observed", fontsize = 16 )
    ax[1].plot(decomposition.trend)
    ax[1].set_title("Trend", fontsize = 16)
    ax[2].plot(decomposition.seasonal)
    ax[2].set_title("Seasonal", fontsize = 16)
    ax[3].plot(decomposition.resid)
    ax[3].set_title("Residuals", fontsize = 16)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR/"Decomposition Plots.png", dpi = 300)

def optimal_sarima():
    sarima_model =  SARIMAX(endog = daily_rides_transformed, order = ORDER, simple_differencing = False).fit()
    fig = sarima_model.plot_diagnostics(figsize = (15,12))
    fig.suptitle("SARIMA Model Diagnostics", fontsize=16)
    fig.savefig(OUTPUT_DIR/"SARIMA Diagnostics.png", dpi = 300)

    residual_sarima = sarima_model.resid
    return acorr_ljungbox(residual_sarima, np.arange(1, 11, 1), return_df = True)

def optimal_sarimax(): 
    sarimax_model =  SARIMAX(endog = ENDOG, exog = EXOG, order = ORDER, seasonal_order= SEASONAL_ORDER, simple_differencing = False).fit()
    fig = sarimax_model.plot_diagnostics(figsize = (15,12))
    fig.suptitle("SARIMAX Model Diagnostics", fontsize=16)
    fig.savefig(OUTPUT_DIR/"SARIMAX Diagnostics.png", dpi = 300)

    residual_sarima = sarimax_model.resid
    return acorr_ljungbox(residual_sarima, np.arange(1, 11, 1), return_df = True)

def rolling_forecast(exogenous = False) -> pd.Series : 
    forecasts_rolling  = pd.Series()
    for end_date in TEST.index:
        endog_train_data = ENDOG[:end_date - timedelta(days = 1)]
        exog_train_data = EXOG.loc[:end_date - timedelta(days = 1)]
        if exogenous == False: 
            fitted_model = SARIMAX(endog_train_data, order = ORDER, seasonal_order = SEASONAL_ORDER).fit()
        else: 
            fitted_model = SARIMAX(endog_train_data, exog_train_data, order = ORDER, seasonal_order = SEASONAL_ORDER).fit()
        pred = fitted_model.forecast(1, exog = EXOG.loc[[end_date]] if exogenous == True else None)

        forecasts_rolling.loc[end_date] = pred.loc[end_date]
    return forecasts_rolling

def inverse_transform(series):
    mu = np.mean(daily_rides)
    sdev = np.std(daily_rides)
    inverse = pd.Series(data = (pt.inverse_transform(series.to_frame())).ravel(), index = series.index )
    inverse = (inverse*sdev) + mu 
    return inverse 

def evaluate(predicted, model_type):
    inverse_predicted = inverse_transform(predicted)
    print(f'\n\n{model_type} Evaluation Metrics :')
    print(f'MAE : {mean_absolute_error(TEST, inverse_predicted)}')
    print(f'MAPE : {mean_absolute_percentage_error(TEST, inverse_predicted)}\n')

def rolling_forecast_plot(forecasts_rolling, model_type):
    fig, ax = plt.subplots(1, figsize = (14,8))
    ax.plot(forecasts_rolling, label = "predicted")
    ax.plot(daily_rides_transformed.iloc[600:], label = 'actual')
    plt.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalised Demand')           
    ax.set_title(f"{model_type} Forecasts", fontsize = 16)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR/f"{model_type} Forecasts.png", dpi = 300)

def baseline_comparison(**kwargs): 
    naive = []
    mean_forecast = pd.Series(
    [daily_rides_transformed.loc[daily_rides.index < date].mean()
     for date in TEST.index],
    index=TEST.index)

    naive_seasonal_forecast = daily_rides_transformed.shift(7).loc[TEST.index]
    previous_day_naive_forecast = daily_rides_transformed.shift(1).loc[TEST.index]

    naive.extend((mean_forecast, naive_seasonal_forecast, previous_day_naive_forecast, *kwargs.values()))
    baseline_stats = [mean_absolute_percentage_error(TEST, inverse_transform(i))*100 for i in naive]

    fig, ax = plt.subplots(figsize = (15,8))
    bar = ax.bar(['Mean Forecast','Naive Seasonal Forecast', "Previous Day Forecast", *[key.replace('_', ' ') for key in kwargs]], baseline_stats)
    ax.bar_label(bar, fmt="%.2f%%")
    ax.set_title('Forecast MAPE', fontsize = 16)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR/"Baseline Comparisons.png")

def initial_plots():
    eda_plots()
    correlelogram_plots()
    decomposition_plot()

def sarima_pipeline(from_saved):
    optimal_sarima()
    if from_saved == True :
        sarima_predictions = pd.read_pickle(OUTPUT_DIR / "sarima_predictions.pkl")
    else : 
        sarima_predictions = rolling_forecast(exogenous = False)
        sarima_predictions.to_pickle(OUTPUT_DIR / "sarima_predictions.pkl")
    evaluate(sarima_predictions, model_type = "SARIMA")
    rolling_forecast_plot(sarima_predictions, model_type = "SARIMA")
    baseline_comparison(Sarima_Forecast = sarima_predictions)
    return sarima_predictions

def sarimax_pipeline(sarima_predictions, from_saved): 
    optimal_sarimax()
    if from_saved == True :
            sarimax_predictions = pd.read_pickle(OUTPUT_DIR / "sarimax_predictions.pkl")
    else : 
        sarimax_predictions = rolling_forecast(exogenous = True)
        sarimax_predictions.to_pickle(OUTPUT_DIR / "sarimax_predictions.pkl")
    evaluate(sarimax_predictions, model_type = "SARIMAX")
    rolling_forecast_plot(sarimax_predictions, model_type = "SARIMAX")
    baseline_comparison(Sarima_Forecast = sarima_predictions, Sarimax_Forecast = sarimax_predictions )
    
def analysis_pipeline(from_saved = True):
    initialise_data()
    initial_plots()
    if from_saved == True:
        sarima_predictions = sarima_pipeline(from_saved = True)
        sarimax_pipeline(sarima_predictions, from_saved = True)
    else : 
        sarima_predictions = sarima_pipeline(from_saved = False)
        sarimax_pipeline(sarima_predictions, from_saved = False)
    print("Evaluation Complete")



