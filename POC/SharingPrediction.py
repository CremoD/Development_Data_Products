import warnings
warnings.filterwarnings("ignore")
import requests
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from tensorflow import keras
import math
from matplotlib import pyplot as plt
import seaborn as sns

# produce plots
def prediction_plots(df, station):
	fig = plt.figure(figsize=(16,15))
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	i = 1
	l = len(df['day'].unique())

	for day in df['day'].unique():
		ax = fig.add_subplot(math.ceil(5/2), 2, i)
		day_plot(df, day, station)
		i +=1

	fig.savefig('plots/prediction.png', bbox_inches = 'tight', pad_inches = 0, dpi=80)


# plot of one day
def day_plot(df, day, station):
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
	df_grouped = df[(df['StartStation Name']== station) &(df['day']== day)]
	ax = sns.lineplot(data=df_grouped, x="hour", y="bike_prediction")
	ax.set_title(station + " predicted bike sharing: " + str(day) + " of " + months[df_grouped['month'].values[0] -1])
	ax.set(xticks=df_grouped['hour'].values)

# function to create the dataframe of forecasts
def create_forecast_df(response_json, stations):
	forecasts_rows = []

	for elem in response_json:
		curr_row = []

		dt_iso = elem['dt_txt']
		temp = round(elem['main']['temp']-273.15, 2)
		feels_like = round(elem['main']['feels_like']-273.15, 2)
		temp_min = round(elem['main']['temp_min']-273.15, 2)
		temp_max = round(elem['main']['temp']-273.15, 2)
		hum = elem['main']['humidity']
		wind = elem['wind']['speed']
		weather = elem['weather'][0]['main']
		curr_row.append(dt_iso)
		curr_row.append(temp)
		curr_row.append(feels_like)
		curr_row.append(temp_min)
		curr_row.append(temp_max)
		curr_row.append(hum)
		curr_row.append(wind)
		curr_row.append(weather)

		#iterate over stations
		for station in stations:
			temp_row = curr_row[:]
			temp_row.insert(1, station)
			#append current row
			forecasts_rows.append(temp_row)

	forecast_df = pd.DataFrame(forecasts_rows, columns =['start_date', 'StartStation Name', 'temp', 'feels_like', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'weather_main'])
	return forecast_df


# function to extract features from date
def extract_from_date(df):

	df['start_date'] = pd.to_datetime(df['start_date'], format ="%Y-%m-%d %H:%M:%S")

	# extract year
	df['year'] = df['start_date'].dt.year
	# extract month
	df['month'] = df['start_date'].dt.month
	# extract day
	df['day'] = df['start_date'].dt.day
	# extract day_of_week (0 monday - 6 sunday)
	df['day_of_week']=df['start_date'].dt.dayofweek
	# extract hour
	df['hour']= df['start_date'].dt.hour

	# extract workday boolean
	df['day_type'] = np.where(df['day_of_week']<5, 'workday', 'weekend')

	# extract season
	conditions_weather = [
		df['month'].isin([1,2,12]),
		df['month'].isin([3,4,5]),
		df['month'].isin([6,7,8]),
		df['month'].isin([9,10,11])
	]
	choices = ['winter', 'spring', 'summer', 'fall']
	df['season'] = np.select(conditions_weather, choices)

	# extract day period
	conditions_period = [
		df['hour'].isin([0,1,2,3,4,5]),
		df['hour'].isin([6,7,8,9,10,11]),
		df['hour'].isin([12,13,14,15,16,17]),
		df['hour'].isin([18,19,20,21,22,23])
	]
	choices = ['night', 'morning', 'afternoon', 'evening']
	df['day_period'] = np.select(conditions_period, choices)

	return df

# one hot encoding
def one_hot(df, ohe):
	l_object = df[["StartStation Name", "weather_main", "season", "day_type", "day_period","day_of_week"]]
	codes = ohe.transform(l_object).toarray()
	feature_names = ohe.get_feature_names(["StartStation Name", "weather_main", "season", "day_type", "day_period","day_of_week"])
	new_df = pd.concat([df[['start_date', 'temp', 'feels_like', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'year', 'month', 'day', 'hour']].reset_index(drop=True), pd.DataFrame(codes,columns=feature_names).reset_index(drop=True)], axis=1)
	return new_df


# handle cyclic features
def cyclic_features(df):
	df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
	df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

	df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
	df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# feature engineering to perform the previously defined functions
def feature_engineering(df, ohe, scaler):
	df_processed = one_hot(df, ohe)
	cyclic_features(df_processed)
	df_processed = df_processed.drop(columns=['start_date', 'year', 'month', 'day', 'hour'])
	df_processed[["feels_like", "humidity", "temp", "temp_max", "temp_min", "wind_speed"]] = scaler.transform(df[["feels_like", "humidity", "temp", "temp_max", "temp_min", "wind_speed"]])
	return df_processed


# predict bike sharing
def predict(df, forecast_df):
	reconstructed_model = keras.models.load_model("dnn_regr")
	y_pred = reconstructed_model.predict(df)
	y_pred = [int(elem) for elem in y_pred]
	forecast_df['bike_prediction'] = y_pred


# combine all the functions and create forecast dataframe with bike sharing predicted
def forecasts_creation(stations):
	with open('encoder/encoder.pickle', 'rb') as f:
		ohe = pickle.load(f)

	with open('encoder/scaler.pickle', 'rb') as f:
		scaler = pickle.load(f)


	api_key = "8b68074189987f8b317ecb78546e0171"
	location = "London"
	url_string = "http://api.openweathermap.org/data/2.5/forecast?q="+location+"&appid=" + api_key;


	response = requests.get(url_string)
	response_json = response.json()['list']


	forecast_df = create_forecast_df(response_json, stations)
	forecast_processed = extract_from_date(forecast_df)
	forecast_processed = feature_engineering(forecast_processed, ohe, scaler)
	predict(forecast_processed, forecast_df)

	return forecast_df