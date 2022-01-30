# Import data_preprocess.py
from ml_models.py_files import data_preprocess

# Import modules
from datetime import date, datetime
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error


def arima(col):
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
		train = df[(df['DATE_TIME'].dt.date >= train_start_date) & (df['DATE_TIME'].dt.date < d)].set_index('DATE_TIME').dropna(subset = [col])

		# Get test set	
		test = df[(df['DATE_TIME'].dt.date == d)].set_index('DATE_TIME').dropna(subset = [col])

		# Exogneous factors (month encoding not considered as the train and test set have same month for all values)
		exog_train = train[['DAY_sin', 'DAY_cos', 'HOUR_sin', 'HOUR_cos']]
		exog_test = test[['DAY_sin', 'DAY_cos', 'HOUR_sin', 'HOUR_cos']]

		# auto_arima model training and predictions
		model = pm.auto_arima(y = train[col], X = exog_train, start_p = 1, start_q = 1, test_g = 'adf', seasonal = False).fit(train[col])
		pred = np.array(model.predict(n_periods = len(test), X = exog_test))
		test['PREDICTED_' + col + '_ARIMA'] = pred

		# Calculate MAPE
		mape_value.append(mean_absolute_percentage_error(test[col], test['PREDICTED_' + col + '_ARIMA']))

		# Append results
		test_filtered = test[[col, 'PREDICTED_' + col + '_ARIMA']]
		predictions = predictions.append(test_filtered)
		mape_list.append(mape_value)

	# Convert mape_list to df and calculate avergae mape for test set
	mape_df = pd.DataFrame(mape_list, columns = [col], index = test_dates)
	mape_df.loc['MAPE'] = mape_df.mean()

	# Return mape dataframe and predictions dataframe
	return mape_df, predictions


def arima_predictions_all():
	# Get CO and TEMP predicitons using auto_arima
	mape_co, predictions_co = arima('CO')
	mape_temp, predictions_temp = arima('TEMP')

	# Write predictions to excel
	with pd.ExcelWriter('Predictions_AutoArima.xlsx') as writer:  
		predictions_co.to_excel(writer, sheet_name = 'CO_AutoArima')
		predictions_temp.to_excel(writer, sheet_name = 'Temp_AutoArima')

	return mape_co, mape_temp, predictions_co, predictions_temp
