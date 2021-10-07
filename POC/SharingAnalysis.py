import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Extract from date
def extract_from_date(df):
	df['start_date'] = pd.to_datetime(df['start_date'], format ="%d/%m/%Y %H:%M")
	
	# extract year
	df['year'] = df['start_date'].dt.year
	# extract month
	df['month'] = df['start_date'].dt.month_name()
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
		df['month'].isin(['January', 'February', 'December']),
		df['month'].isin(['March','April','May']),
		df['month'].isin(['June','July','August']),
		df['month'].isin(['September','October','November'])
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


# extract from weather
def extract_from_weather(df):
	# extract temp range
	conditions_temp = [
		df['temp']<-10.0,
		(df['temp']>=-10.0) & (df['temp']<0.0),
		(df['temp']>=0.0) & (df['temp']<10.0),
		(df['temp']>=10.0) & (df['temp']<20.0),
		(df['temp']>=20.0) & (df['temp']<30.0),
		(df['temp']>=30.0) & (df['temp']<40.0),
		df['temp']>40.0
	]
	choices = ['< -10°C', '-10°C - 0°C', '0°C - 10°C', '10°C - 20°C', '20°C - 30°C', '30°C - 40°C', '> 40°C']
	df['temp_range'] = np.select(conditions_temp, choices)
    
	# extract wind range                
	conditions_wind = [
		(df['wind_speed']>=0.0) & (df['wind_speed']<5.0),
		(df['wind_speed']>=5.0) & (df['wind_speed']<10.0),
		(df['wind_speed']>=10.0) & (df['wind_speed']<15.0),
		(df['wind_speed']>=15.0) & (df['wind_speed']<20.0),
		(df['wind_speed']>=20.0) & (df['wind_speed']<25.0),
		df['wind_speed']>=25
	]
	choices = ['0-5 m/s', '5-10 m/s', '10-15 m/s', '15-20 m/s', '20-25 m/s', '> 25 m/s']
	df['wind_range'] = np.select(conditions_wind, choices)
    
	return df


 # extract features
def extract_features(df):
	curr_df = extract_from_date(df)
	final_df = extract_from_weather(curr_df)
	return final_df


# consider general bike sharing usage
def total_shares(df):
	total_shares = df.groupby(['start_date', 'temp', 'feels_like','temp_min', 'temp_max','humidity','wind_speed','weather_main','year','month','day_of_week','hour','day_type','season','day_period', 'day', 'temp_range', 'wind_range']).sum().reset_index()
	return total_shares


############################ Produce different plots ############################
# bikes rented by year
def year_plot(df, axx):
	df_grouped = df.groupby('year').mean().reset_index()
	df_grouped['n_bike_rented'] = df_grouped['n_bike_rented']*24
	plt.figure(figsize=(10,6))
	ax = sns.barplot(x="year", y = "n_bike_rented", data=df_grouped, ax = axx)
	ax.set_title("Average number of bikes rented daily over the years")
	return ax

# bikes rented per month
def month_plot(df,axx):
	df_grouped = df.groupby('month').mean().reset_index()
	df_grouped['n_bike_rented'] = df_grouped['n_bike_rented']*24
	plt.figure(figsize=(10,6))
	plot_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']
	ax = sns.barplot(x="month", y = "n_bike_rented", data=df_grouped, palette="deep", order=plot_order, ax = axx)
	ax.set_title("Average number of bikes rented daily over the months")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
	return ax

# bikes rented per season
def season_plot(df, axx):
	df_grouped = df.groupby('season').mean().reset_index()
	df_grouped['n_bike_rented'] = df_grouped['n_bike_rented']*24
	plt.figure(figsize=(10,6))
	plot_order = ['winter', 'spring', 'summer', 'fall']
	ax = sns.barplot(x="season", y = "n_bike_rented", data=df_grouped, order = plot_order, ax = axx)
	ax.set_title("Average number of bikes rented daily over the seasons")
	return ax

# bikes rented per day type
def day_type_plot(df, axx):
	df_grouped = df.groupby('day_type').mean().reset_index()
	df_grouped['n_bike_rented'] = df_grouped['n_bike_rented']*24
	plt.figure(figsize=(10,6))
	ax = sns.barplot(x="day_type", y = "n_bike_rented", data=df_grouped, ax = axx)
	ax.set_title("Average number of bikes rented daily by type of the day")
	return ax

# bikes rented per day period
def day_period_plot(df, axx):
	df_grouped = df.groupby(['day_type', 'day_period']).mean().reset_index()
	df_grouped['n_bike_rented'] = df_grouped['n_bike_rented']*6
	plt.figure(figsize=(10,6))
	plot_order = ['morning', 'afternoon', 'evening', 'night']
	ax = sns.barplot(x='day_period', y="n_bike_rented", hue="day_type", order = plot_order, data=df_grouped, ax = axx)
	ax.set_title("Average number of bikes rented daily by day period and type")
	return ax

# top 5 hours
def top_5_hours(df, axx):
	df_grouped = df.groupby('hour').mean().reset_index()
	plt.figure(figsize=(10,6))
	df_top = df_grouped.nlargest(5, 'n_bike_rented')
	plot_order = df_top.sort_values(by='n_bike_rented', ascending=False).hour.values
	ax = sns.barplot(data=df_top, x="hour", y="n_bike_rented", order = plot_order, ax = axx)
	ax.set_title("Top 5 hours when bikes are more requested")
	return ax

# hourly distribution
def hour_plot(df, axx):
	df_grouped = df.groupby('hour').mean().reset_index()
	plt.figure(figsize=(10,6))
	ax = sns.lineplot(data=df_grouped, x="hour", y="n_bike_rented", ax = axx)
	ax.set_title("Hourly distribution of bikes rented")
	ax.set(xticks=df_grouped['hour'].values)
	return ax

# hourly distribution and day type
def hour_daytype_plot(df, axx):
	df_grouped = df.groupby(['hour', 'day_type']).mean().reset_index()
	plt.figure(figsize=(10,6))
	ax = sns.lineplot(data=df_grouped, x="hour", y="n_bike_rented", hue = 'day_type', ax = axx)
	ax.set_title("Hourly distribution of bikes rented, by type of the day")
	ax.set(xticks=df_grouped['hour'].values)
	return ax

# top 10 stations
def stations_plot(df, axx):
	df_grouped = df.groupby('StartStation Name').sum().reset_index()
	plt.figure(figsize=(10,6))
	ax = sns.barplot(data=df_grouped.nlargest(10, 'n_bike_rented'), x="StartStation Name", y="n_bike_rented", ax = axx)
	ax.set_title("Top 10 most used stations")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
	return ax

# weather plot
def weather_plot(df, axx):
	df_grouped = df.groupby('weather_main').mean().reset_index()
	plt.figure(figsize=(10,6))
	ax = sns.barplot(data=df_grouped, y="weather_main", x="n_bike_rented", ax = axx)
	ax.set_title("Average number of bikes rented hourly by weather condition")
	return ax

# wind plot
def wind_plot(df, axx):
	df_grouped = df.groupby('wind_range').mean().reset_index()
	plt.figure(figsize=(10,6))
	plot_order = ['0-5 m/s', '5-10 m/s', '10-15 m/s', '15-20 m/s', '20-25 m/s', '> 25 m/s']
	ax = sns.barplot(data=df_grouped, x="wind_range", y="n_bike_rented", order = plot_order, ax = axx)
	ax.set_title("Average number of bikes rented hourly by wind speed")
	return ax

# temp plot
def temp_plot(df,axx):
	df_grouped = df.groupby('temp_range').mean().reset_index()
	plt.figure(figsize=(10,6))
	plot_order = ['< -10°C', '-10°C - 0°C', '0°C - 10°C', '10°C - 20°C', '20°C - 30°C', '30°C - 40°C', '> 40°C']
	ax = sns.barplot(data=df_grouped, x="temp_range", y="n_bike_rented", order = plot_order, ax = axx)
	ax.set_title("Average number of bikes rented hourly by temp")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
	return ax

# total bike shares per hour grid
def total_grid(df, df_stations):
	fig = plt.figure(figsize=(16,35))
	fig.subplots_adjust(hspace=0.4, wspace=0.3)
	ax = fig.add_subplot(6, 2, 1)
	year_plot(df,ax)
	ax = fig.add_subplot(6, 2, 2)
	month_plot(df,ax)
	ax = fig.add_subplot(6, 2, 3)
	season_plot(df,ax)
	ax = fig.add_subplot(6, 2, 4)
	day_type_plot(df,ax)
	ax = fig.add_subplot(6, 2, 5)
	day_period_plot(df,ax)
	ax = fig.add_subplot(6, 2, 6)
	top_5_hours(df,ax)
	ax = fig.add_subplot(6, 2, 7)
	hour_plot(df,ax)
	ax = fig.add_subplot(6, 2, 8)
	hour_daytype_plot(df,ax)
	ax = fig.add_subplot(6, 2, 9)
	stations_plot(df_stations,ax)
	ax = fig.add_subplot(6, 2, 10)
	weather_plot(df,ax)
	ax = fig.add_subplot(6, 2, 11)
	wind_plot(df,ax)
	ax = fig.add_subplot(6, 2, 12)
	temp_plot(df,ax)

	fig.savefig('plots/analysis.png', bbox_inches = 'tight', pad_inches = 0, dpi=80)

# specific bike shares per station
def station_grid(df):
	fig = plt.figure(figsize=(16,35))
	fig.subplots_adjust(hspace=0.4, wspace=0.3)
	ax = fig.add_subplot(6, 2, 1)
	year_plot(df,ax)
	ax = fig.add_subplot(6, 2, 2)
	month_plot(df,ax)
	ax = fig.add_subplot(6, 2, 3)
	season_plot(df,ax)
	ax = fig.add_subplot(6, 2, 4)
	day_type_plot(df,ax)
	ax = fig.add_subplot(6, 2, 5)
	day_period_plot(df,ax)
	ax = fig.add_subplot(6, 2, 6)
	top_5_hours(df,ax)
	ax = fig.add_subplot(6, 2, 7)
	hour_plot(df,ax)
	ax = fig.add_subplot(6, 2, 8)
	hour_daytype_plot(df,ax)
	ax = fig.add_subplot(6, 2, 9)
	weather_plot(df,ax)
	ax = fig.add_subplot(6, 2, 10)
	wind_plot(df,ax)
	ax = fig.add_subplot(6, 2, 11)
	temp_plot(df,ax)

	fig.savefig('plots/analysis.png', bbox_inches = 'tight', pad_inches = 0, dpi=80)


# Final grid of plots. Filtered df is True if we are looking for specific station (one plot should be removed)
def grid_plots(filtered_df):
	london_df = pd.read_csv('weather_bike_usage.csv')
	london_processed = extract_features(london_df)

	if filtered_df == 'total':
		df = total_shares(london_processed)
		fig = total_grid(df, london_processed)
	else:
		df = london_processed[london_processed['StartStation Name'] == filtered_df]
		fig = station_grid(df)

	return fig

