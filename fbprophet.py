# Import data_preprocess.py
import data_preprocess

# Import modules
from datetime import date, datetime
import pandas as pd
import numpy as np
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


def fbprophet(cols, param):
	# Get data from data_preprocess.py
	df = data_preprocess.preprocess()

	# Set train and test dates
	train_start_date = date(2004, 3, 11)

	test_dates_datetime = pd.date_range(start = '2004-03-18', end = '2004-03-24')
	test_dates_string = [d.strftime('%Y-%m-%d') for d in test_dates_datetime]
	test_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in test_dates_string]
	
	# Create empty dataframe to store predictions
	predictions = pd.DataFrame()
	# Create empty list to store mape values
	mape_list = []

	# Iterate for all test dates
	for d in test_dates:
		mape_value = []

		# Get train set
		train_df = df[(df['DATE_TIME'].dt.date >= train_start_date) & (df['DATE_TIME'].dt.date < d)].sort_values(by = ['DATE_TIME'])
		# Get test set	
		test = df[(df['DATE_TIME'].dt.date == d)].reset_index(drop = True)

		# Iterate through the feature engineered columns and make predictions using each of them
		for col in cols:
			# Creat empty df to combine all predictions
			output_df = pd.DataFrame()
			# Filter columns  for training FBProphet
			train = train_df[['DATE_TIME', col, 'DAY_sin', 'DAY_cos', 'HOUR_sin', 'HOUR_cos']]
			# Rename columns as required by model
			train.rename(columns = {'DATE_TIME' : 'ds', col : 'y'}, inplace = True)

			# Train prophet model and make predictions
			model = Prophet()
			# Add holidays
			model.add_country_holidays(country_name = 'Italy')
			# Add cyclical data as regressors 
			model.add_regressor('DAY_sin')
			model.add_regressor('DAY_cos')
			model.add_regressor('HOUR_sin')
			model.add_regressor('HOUR_cos')
			model.fit(train)

			# Create future daatframe
			future = model.make_future_dataframe(periods = len(test), freq = 'H', include_history = False)
			# Add regressors as columns to future dataframes
			future['DAY_sin'] = test['DAY_sin']
			future['DAY_cos'] = test['DAY_cos']
			future['HOUR_sin'] = test['HOUR_sin']
			future['HOUR_cos'] = test['HOUR_cos']

			# Predict and calculate MAPE
			forecast = model.predict(future)
			mape_values.append(mean_absolute_percentage_error(test[col], forecast.yhat))

			# Generate output df
			output_df = test[['DATE_TIME', param]].reset_index(drop = True)
			forecast_df = forecast[['yhat']]
			forecast_df.rename(columns = {'yhat' : 'PREDICTED_' + param + '_PROPHET'}, inplace = True)
			joined_output = pd.concat([output_df, forecast_df], axis = 1)
			joined_output['METHOD'] = col

			# Merge predictions
			predictions_co = predictions_co.append(joined_output)

		# Append MAPE values
		mape_list.append(mape_values)

		# Create df from mape_list
		mape_df = pd.DataFrame(mape_list, columns = cols, index = test_dates)
		mape_df.loc['MAPE'] = mape_df.mean()

	return mape_df, predicitons

def fbprophet_predictions_all():
	cols_co = ['LIN_INTERPOLATE_CO', 'BFILL_CO', 'FFILL_CO', 'SMA_CO_1_DAY', 'SMA_CO_7_DAY', 'AVG_CO']
	cols_temp = ['TEMPERATURE', 'LIN_INTERPOLATE_T', 'BFILL_T', 'FFILL_T', 'SMA_T_1_DAY', 'SMA_T_7_DAY', 'AVG_T']

	# Get CO and TEMP predicitons using fbprophet
	mape_co, predictions_co = fbprophet(cols_co, 'CO')
	mape_temp, predictions_temp = arima(cols_temp, 'TEMP')

	# Concat mape 
	mape_concat_prophet = pd.concat([mape_temp, mape_co], axis = 1)

	# return mape df, CO predictions and TEMP predictions
	return mape_concat_prophet, predictions_co, predictions_temp
