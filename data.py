import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
import holidays
from datetime import datetime 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acf,adfuller, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL 
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.preprocessing import PowerTransformer 
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error 
from scipy.stats import yeojohnson, boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

import requests
from pathlib import Path

import pandas as pd
import requests
from pathlib import Path

#Downloads the full dataset for period 2024-10-01 - 2026-08-31 for later processing 
def get_data():
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    folder = Path("yellow_taxi_data")
    folder.mkdir(exist_ok=True)

    months = [f'{month.year}-{month.month:02d}' for month in pd.date_range(start = '2024-10-01', end = '2026-08-31', freq = 'ME')]
    for month in months:
        filename = f"yellow_tripdata_{month}.parquet"
        filepath = folder / filename
        
        if filepath.exists():
            print("Already downloaded:", filename)
            continue
        url = base_url + filename

        response = requests.get(url, stream=True, timeout=60)
    
        if response.status_code == 200:
            with open(folder / filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            print("Downloaded:", filename)
        else:
            print("Unavailable:", filename)

        response.close()

#Builds the timeseries by selecting relevant data 
def select_daily_demand():
    files = glob("yellow_taxi_data/*.parquet")

    df = pd.concat(
        [pd.read_parquet(f, columns=["tpep_pickup_datetime"]) for f in files],
        ignore_index=True
    )
    df['date'] = df["tpep_pickup_datetime"].dt.date
    daily_rides = (df.groupby("date").size())
    daily_rides.index = pd.to_datetime(daily_rides.index)
    daily_rides = daily_rides.asfreq('D')
    daily_rides = daily_rides[(datetime(2024,10,1) <= daily_rides.index) & (daily_rides.index<= datetime(2026,8,31))]
    return daily_rides


def normalise_series(daily_rides):
    daily_rides_normalised = (daily_rides - daily_rides.mean()) / daily_rides.std()
    return daily_rides_normalised

def transform_series(daily_rides_normalised):
    pt = PowerTransformer(method = 'yeo-johnson', standardize = False)
    transformed = pt.fit_transform(daily_rides_normalised.to_frame())
    daily_rides_transformed = pd.Series(data = transformed.ravel(), index =  daily_rides_normalised.index)
    return daily_rides_transformed, pt 

def build_master_data(daily_rides_transformed):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": f'{daily_rides_transformed.index.date[0]}',
        "end_date": f'{daily_rides_transformed.index.date[699]}',
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "precipitation_sum",
            "snowfall_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    weather = pd.DataFrame(response.json()["daily"])

    weather.index = daily_rides_transformed.index
    weather.drop(columns = 'time', inplace = True)

    us_holidays = holidays.US(years = [2024,2025,2026])
    start, end = daily_rides_transformed.index.date[0], daily_rides_transformed.index.date[699] 
    
    selected_us_holidays = {
            d : name for d,name in us_holidays.items() if start <= d <= end
        }

    holiday_keys = [pd.to_datetime(i) for i in selected_us_holidays.keys()]

    master_data = pd.concat([daily_rides_transformed, weather], ignore_index = False, axis = 1)
    master_data['holiday_day'] = np.full(700,0)

    for i in holiday_keys: 
        master_data.loc[i] = 1

    return master_data

def build_final_df():
    daily_rides = select_daily_demand()
    daily_rides_normalised = transform_series(daily_rides)
    daily_rides_transformed, pt = transform_series(daily_rides_normalised)
    return build_master_data(daily_rides_transformed)

def get_transformed_series():
    daily_rides = select_daily_demand()
    daily_rides_normalised = normalise_series(daily_rides)
    daily_rides_transformed, pt = transform_series(daily_rides_normalised)
    return daily_rides_transformed, pt

