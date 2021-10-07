import time
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Read a csv file and clean it by processing start date and stations
def read_clean_dataset(path):
    
    # read dataset
    df = pd.read_csv(path, encoding='cp1252')
    
    # preprocess start date and station name
    df['Start Date'] = df['Start Date'].replace('(:\d\d:\d\d)|(:\d\d)',':00',regex=True)  
    df['StartStation Name'] = df['StartStation Name'].replace('.*[,:]\s?','',regex=True)
    
    # cleaning by removing test data
    df = df[(df['StartStation Name'] != "PENTON STREET COMMS TEST TERMINAL _ CONTACT MATT McNULTY") &
            (df['StartStation Name'] != "1") &
            (df['StartStation Name'] != "0") &
            (df['StartStation Name'] != "Pop Up Dock 1") &
            (df['StartStation Name'] != "tabletop1") &
            (df['StartStation Name'] != "Guildhall (REMOVED)")]
    
    # group by start station name and start date
    df = df.groupby(['Start Date', 'StartStation Name']).size().reset_index(name='n_bike_rented')
    return df

# New function



###########################################################################
# Read the various csv files of different length and put them together 
directory = "usage/"
i = 1
# first dataset
df = read_clean_dataset("usage/1. Journey Data Extract 04Jan-31Jan 12.csv")
#df['StartStation Name'] = df['StartStation Name'].replace(':',',',regex=True)


for filename in os.listdir(directory):
    print("{0}: {1}".format(i, filename))
    i += 1
    if filename != '1. Journey Data Extract 04Jan-31Jan 12.csv':
        curr_df = read_clean_dataset(directory + filename)
        #curr_df['StartStation Name'] = curr_df['StartStation Name'].replace(':',',',regex=True)
        df = df.append(curr_df, sort = False)

###########################################################################
# Read and process weather dataset

weather_dataset = pd.read_csv('weather/london_history_weather.csv')
# select columns of interest
weather_dataset = weather_dataset[['dt_iso', 'temp', 'feels_like', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'weather_main']]
# clean dt_iso to match with rental_usage dataset
weather_dataset['dt_iso'] = weather_dataset['dt_iso'].replace(':\d\d\s\+.*UTC','',regex=True)  
weather_dataset['dt_iso'] = weather_dataset['dt_iso'].replace('-','/',regex=True)  
weather_dataset = weather_dataset.drop_duplicates(subset=['dt_iso'])

# join two datasets
enriched_df = df.set_index('Start Date').join(weather_dataset.set_index('dt_iso'))
enriched_df = enriched_df.reset_index().rename(columns={"index": "start_date"})
enriched_df.to_csv("weather_bike_usage.csv", index=False)
