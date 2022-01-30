# Import data_preprocess.py
from baseline_models.py_files import data_preprocess

# Import modules
from datetime import date, datetime
import pandas as pd
# from fbprophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def fbprophet(cols, param):
	# Get data from data_preprocess.py
	df = data_preprocess.preprocess()

	# Set train and test dates
	train_start_date = date(2004, 3, 11)

	test_dates_datetime = pd.date_range(start = '2004-03-18', end = '2004-03-24')
	test_dates_string = [d.strftime('%Y-%m-%d') for d in test_dates_datetime]
	test_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in test_dates_string]
	
	# Create dataframe to store predictions
	predictions = pd.DataFrame()

	# Create empty list to store mape values
	mape_list = []

	# Iterate for all test dates
	for d in test_dates:
		mape_values = []

		# Get train set and test set
		test = df[(df['DATE_TIME'].dt.date == d)].reset_index(drop = True)
		train_df = df[(df['DATE_TIME'].dt.date >= train_start_date) & (df['DATE_TIME'].dt.date < d)].sort_values(by = ['DATE_TIME']).dropna(subset = [param])
		
		output_df = test[['DATE_TIME', param]].reset_index(drop = True).dropna(subset = [param])

		# Iterate through the feature engineered columns and make predictions using each of them
		for col in cols:
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
			output_df['PREDICTED_' + col + 'FBPROPHET'] = forecast.yhat
			predictions = predictions.append(output_df)

		# Drop NaN's resulting from append operation
		predictions = predictions.dropna()

		# Append MAPE values
		mape_list.append(mape_values)

		# Create df from mape_list
		mape_df = pd.DataFrame(mape_list, columns = cols, index = test_dates)
		mape_df.loc['MAPE'] = mape_df.mean()

	return mape_df, predictions

def fbprophet_predictions_all():
	cols_co = ['LIN_INTERPOLATE_CO', 'BFILL_CO', 'FFILL_CO', 'SMA_CO_1_DAY', 'SMA_CO_7_DAY', 'AVG_CO']
	cols_temp = ['TEMPERATURE']

	# Get CO and TEMP predicitons using fbprophet
	mape_co, predictions_co = fbprophet(cols_co, 'CO')
	mape_temp, predictions_temp = fbprophet(cols_temp, 'TEMP')

	# Write to excel
	with pd.ExcelWriter('Predictions_FbProphet.xlsx') as writer:  
		predictions_co.to_excel(writer, sheet_name = 'CO_FbProphet')
		predictions_temp.to_excel(writer, sheet_name = 'Temp_FbProphet')

	return mape_co, mape_temp, predictions_co, predictions_temp
