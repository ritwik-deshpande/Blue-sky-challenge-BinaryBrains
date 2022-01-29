# Imports
import sys
import subprocess
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Implement pip as a subprocess to install pmdarima, fbprophet, neuralprophet
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pmdarima'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fbprophet'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'neuralprophet'])

def read_data():
	# Read input file and format
	df = pd.read_csv('../Data/AirQuality.csv', delimiter =';', usecols = ['Date', 'Time', 'CO(GT)', 'T'], decimal =',')
	df = df.rename(columns = {'Date' : 'DATE', 'Time' : 'TIME', 'CO(GT)' : 'CO', 'T' : 'TEMP'})
	return df

def preprocess():
	# Get raw data
	df = read_data()

	# Drop nulls
	df = df.dropna()

	# Convert date column from object to datetime object (format - yyyy-mm-dd)
	df['TIME'] = (df['TIME'].replace('\\.',':', regex = True))
	df.loc[:,'DATE_TIME'] = pd.to_datetime(df.DATE.astype(str) + ' ' + df.TIME.astype(str))

	# Sort by DATE_TIME 
	df = df.sort_values(by = 'DATE_TIME').reset_index(drop = True)

	# Handling missing values marked with -200
	# Replace -200 with NaN
	df.loc[df.CO == -200, 'CO'] =  np.nan
	df.loc[df.TEMP == -200, 'TEMP'] =  np.nan

	# Linear interpolation
	df['LIN_INTERPOLATE_T'] = df.TEMP.interpolate(method = 'linear', axis = 0)
	df['LIN_INTERPOLATE_CO'] = df.CO.interpolate(method = 'linear', axis = 0)

	# Backfill
	df['BFILL_T'] = df.TEMP.bfill(axis = 'rows')
	df['BFILL_CO'] = df.CO.bfill(axis = 'rows')

	# ForwardFill
	df['FFILL_T'] = df.TEMP.ffill(axis = 'rows')
	df['FFILL_CO'] = df.CO.ffill(axis = 'rows')

	# Average of value and hour prior and after, rolling avergae over 1 day and rolling avergae over 7 days for CO
	indexes = df[df['CO'].isnull()].index.tolist()
	df['AVG_CO'] = df['CO']
	df['SMA_CO_1_DAY'] = df['CO']
	df['SMA_CO_7_DAY'] = df['CO']
	for i in indexes:
		if (pd.isna(df.loc[i - 1, 'AVG_CO'])) & (pd.isna(df.loc[i + 1, 'AVG_CO'])):
			df.at[i, 'AVG_CO'] = (df.loc[i - 1, 'AVG_CO'] + df.loc[i + 1, 'AVG_CO']) / 2
		else:
			df.at[i, 'AVG_CO'] = df.loc[i - 1, 'AVG_CO']
		
		df.at[i, 'SMA_CO_1_DAY'] = np.mean(df.loc[i - 24 : i, 'SMA_CO_1_DAY'])
		df.at[i, 'SMA_CO_7_DAY'] = np.mean(df.loc[i - 162 : i, 'SMA_CO_7_DAY'])

	# Average of value and hour prior and after, rolling avergae over 1 day and rolling avergae over 7 days for TEMP
	indexes = df[df['TEMP'].isnull()].index.tolist()
	df['AVG_T'] = df['TEMP']
	df['SMA_T_1_DAY'] = df['TEMP']
	df['SMA_T_7_DAY'] = df['TEMP']
	for i in indexes:
		if (pd.isna(df.loc[i - 1, 'AVG_T'])) & (pd.isna(df.loc[i + 1, 'AVG_T'])): 
			df.at[i, 'AVG_T'] = (df.loc[i-1, 'AVG_T'] + df.loc[i + 1, 'AVG_T']) / 2
		else:
			df.at[i, 'AVG_T'] = df.loc[i - 1, 'AVG_T']

		df.at[i, 'SMA_T_1_DAY'] = np.mean(df.loc[i - 24 : i, 'SMA_T_1_DAY'])
		df.at[i, 'SMA_T_7_DAY'] = np.mean(df.loc[i - 162 : i, 'SMA_T_7_DAY'])

	# Extract Month, Day, Hour from DATE_TIME column
	df['MONTH'] = df['DATE_TIME'].dt.month
	df['DAY'] = df['DATE_TIME'].dt.day
	df['HOUR'] = df['DATE_TIME'].dt.hour 

	# Cyclical encoding for hour, day, month 
	cyclical_cols = {'MONTH' : 12, 'DAY' : 31, 'HOUR' : 23}
	for c, max_val in cyclical_cols.items():
		df[c + '_sin'] = np.sin(2 * np.pi * df[c] / max_val)
		df[c + '_cos'] = np.cos(2 * np.pi * df[c] / max_val)

	# Drop unnecessary columns
	df = df.drop(['DATE', 'TIME', 'MONTH', 'DAY', 'HOUR'], axis = 1)

	# Return preprocessed data
	return df
