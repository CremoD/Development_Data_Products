import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# Extract features from date
def extract_from_date(df):
    
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
    
    return df[df['year'] != 2015]


# Apply transformations to perform feature engineering
def one_hot(df):
	l_object = df[["StartStation Name", "weather_main", "season", "day_type", "day_period","day_of_week"]]
	ohe = OneHotEncoder()
	ohe.fit(l_object)
	codes = ohe.transform(l_object).toarray()
	feature_names = ohe.get_feature_names(["StartStation Name", "weather_main", "season", "day_type", "day_period","day_of_week"])
	df_processed = pd.concat([df[['start_date', 'n_bike_rented', 'temp', 'feels_like', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'year', 'month', 'day', 'hour']].reset_index(drop=True), 
               pd.DataFrame(codes,columns=feature_names).reset_index(drop=True)], axis=1)

	# save one hot encoder for reusing it
	with open('encoder/encoder.pickle', 'wb') as f:
		pickle.dump(ohe, f)

	return df_processed


# handle cyclic features
def cyclic_features(df):
	df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
	df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

	df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
	df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# scaling of numerical features
def scale(df):
	scaler = StandardScaler()
	scaler.fit(df[["feels_like", "humidity", "temp", "temp_max", "temp_min", "wind_speed"]])
	df[["feels_like", "humidity", "temp", "temp_max", "temp_min", "wind_speed"]] = scaler.transform(df[["feels_like", "humidity", "temp", "temp_max", "temp_min", "wind_speed"]])

	# save scaler for reusing it
	with open('encoder/scaler.pickle', 'wb') as f:
		pickle.dump(scaler, f)


# combine feature engineering transformations
def feature_engineering(df):
	df_processed = one_hot(df)
	cyclic_features(df_processed)
	# remove columns
	df_processed = df_processed.drop(columns=['start_date', 'year', 'month', 'day', 'hour'])
	scale(df_processed)

	return df_processed


# build and compile model
def build_and_compile_model():
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim = 174, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# regression
def perform_regression(df):
	y = df['n_bike_rented']
	X = df.iloc[:, 1:]
	# split 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	dnn_model = build_and_compile_model()
	# fit model
	history = dnn_model.fit(X_train, y_train, validation_split=0.2, verbose=True, epochs=20)
	# evaluate model
	test_result = dnn_model.evaluate(X_test, y_test, verbose=1)
	print("Test result = ", test_result)
	test_predictions = dnn_model.predict(X_test).flatten()
	test_predictions = [int(elem) for elem in test_predictions]
	y_true = list(y_test)
	rmse = mean_squared_error(y_true, test_predictions, squared=False)
	print("RMSE", rmse)
	mae = mean_absolute_error(y_true, test_predictions)
	print("MAE", mae)

	# save model
	dnn_model.save('dnn_regr')




london_df = pd.read_csv('weather_bike_usage.csv')
london_df['start_date'] = pd.to_datetime(london_df['start_date'], format ="%d/%m/%Y %H:%M")
london_processed = extract_from_date(london_df)
df_processed = feature_engineering(london_processed)
perform_regression(df_processed)


