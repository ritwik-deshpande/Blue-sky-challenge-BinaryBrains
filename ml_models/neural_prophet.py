# Import data_preprocess.py
from ml_models import data_preprocess

# Import modules
from datetime import date, datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from neuralprophet import NeuralProphet
from sklearn.model_selection import ParameterGrid
import random

def choose_hyperparameters(train_start_date, train_end_date, val_start_date, val_end_date, col, df):
	# Create train set
	train_df = df[(df['DATE_TIME'].dt.date >= train_start_date) & (df['DATE_TIME'].dt.date <= train_end_date)].sort_values(by = ['DATE_TIME']).dropna(subset = [col])
	train_data = train_df[['DATE_TIME', col]]
	train_data.columns = ['ds', 'y']

	# Create validation set to choose hyperparameters
	val_df = df[(df['DATE_TIME'].dt.date >= val_start_date) & (df['DATE_TIME'].dt.date <= val_end_date)].sort_values(by = ['DATE_TIME']).dropna(subset = [col])
	val_data = val_df[['DATE_TIME', col]]
	val_data.columns = ['ds', 'y']

	# Dataframe to save the hyperparameters
	model_parameters = pd.DataFrame(columns = ['MAPE', 'PARAMETERS'])

	# ParameterGrid
	params_grid = {'num_hidden_layers' : (1, 2, 4, 8),
					'learning_rate' : [0.0001, 0.001, 0.01, 0.05, 0.1, 1]}
	grid = ParameterGrid(params_grid)

	# For each combination of parameters
	for p in grid:
		random.seed(0)
		# Train neural prophet net and predict on validation set
		train_model = NeuralProphet(growth = 'off', 
                                  daily_seasonality = True, 
                                  loss_func = 'MAE',
                                  num_hidden_layers = p['num_hidden_layers'],
                                  learning_rate = p['learning_rate'],
                                  normalize = 'off',
                                  seasonality_mode = 'additive')
		train_model.fit(train_data, freq ='H')
		p['epochs'] = train_model.config_train.epochs
		p['batch_size'] = train_model.config_train.batch_size

		forecast = train_model.predict(val_data)

		# Store mape values of validation set
		MAPE =  mean_absolute_percentage_error(forecast.y, forecast.yhat1)
		model_parameters = model_parameters.append({'MAPE' : MAPE, 'PARAMETERS' : p}, ignore_index = True)

	return model_parameters

def predict_neural_prophet(train_start_date, test_dates, model_parameters, df, col):
	# Create empty lists to store MAPE values and forecasts
	mape_list = []
	forecasts = []
	# Create empty dataframe to store predictions
	predictions = pd.DataFrame()

	for i, d in enumerate(test_dates):
		# Get train set
		train_df = df[(df['DATE_TIME'].dt.date >= train_start_date) & (df['DATE_TIME'].dt.date < d)].sort_values(by = ['DATE_TIME']).dropna(subset = [col])
		train = train_df[['DATE_TIME', col]]
		train.columns = ['ds', 'y']

		# Get test set
		test_df = df[(df['DATE_TIME'].dt.date == d)].dropna(subset = [col]).reset_index(drop = True)
		test = test_df[['DATE_TIME', col]]
		test.columns = ['ds', 'y']

		random.seed(0)

		# Extract trianed hyperparameters
		hyperparameters = model_parameters.loc[model_parameters.DATE == d, 'PARAMETERS']
		num_hidden_layers = hyperparameters[i]['num_hidden_layers']
		learning_rate = hyperparameters[i]['learning_rate']
		epochs = hyperparameters[i]['epochs']
		batch_size = hyperparameters[i]['batch_size']

		# Train neural prophet and forecast
		model = NeuralProphet(growth = 'off', 
							daily_seasonality = True,
							loss_func = 'MAE',
							num_hidden_layers = num_hidden_layers,
							learning_rate = learning_rate,
							normalize = 'off',
							seasonality_mode = 'additive',
							epochs = epochs,
							batch_size = batch_size)
		model.fit(train, freq ='H')
		forecast = model.predict(test)
		forecast = forecast.reset_index(drop = True)
		test['PREDICTED_' + col + '_NEURALPROPHET'] = forecast.yhat1

		# Create output df
		test_filtered = test[['ds', 'y', 'PREDICTED_' + col + '_NEURALPROPHET']]
		test_filtered.columns = ['DATE_TIME', col, 'PREDICTED_' + col + '_NEURALPROPHET']
		predictions = predictions.append(test_filtered)

		# Calculate MAPE
		mape =  mean_absolute_percentage_error(test_filtered[col], test_filtered['PREDICTED_' + col + '_NEURALPROPHET'])
		mape_list.append(mape)

	predictions = predictions.reset_index()

	return mape_list, predictions

def neural_prophet(col):
	# Get data from data_preprocess.py
	df = data_preprocess.preprocess()

	# Set train and test dates
	train_start_date = date(2004, 3, 11)

	test_dates_datetime = pd.date_range(start = '2004-03-18', end = '2004-03-24')
	test_dates_string = [d.strftime('%Y-%m-%d') for d in test_dates_datetime]
	test_dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in test_dates_string]

	# Set varaibles
	no_of_days = 7
	train_start = 11
	train_counter = 4
	val_start = 16
	val_counter = 0
	test_start = 18
	test_counter = 0

	# Dataframe to store optimal parameters for each day in test set
	model_parameters = pd.DataFrame(columns = ['DATE', 'PARAMETERS'])

	for day in range(0, no_of_days):
		# Run for all combinations of hyper parameters
		model_tuning = choose_hyperparameters(train_start_date, date(2004, 3, train_start + train_counter), 
                                       date(2004, 3, val_start + val_counter), date(2004, 3, val_start + val_counter + 1), col, df)  

		train_counter += 1
		val_counter += 1

		# Store hyperparameters which gives minimum MAPE on validaiton set
		parameters = model_tuning.sort_values(by = ['MAPE']).reset_index(drop = True)
		model_parameters = model_parameters.append({'DATE' : date(2004, 3, test_start + test_counter), 'PARAMETERS' : parameters['PARAMETERS'][0]}, ignore_index = True)
		test_counter += 1

	mape_list, predictions = predict_neural_prophet(train_start_date, test_dates, model_parameters, df, col)

	# Convert mape_list to df and calculate avergae mape for test set
	mape_df = pd.DataFrame(mape_list, columns = [col], index = test_dates)
	mape_df.loc['MAPE'] = mape_df.mean()

	# Return mape dataframe and predictions dataframe
	return mape_df, predictions

def neural_prophet_predictions_all():
	# Get CO and TEMP predicitons using auto_arima
	mape_co, predictions_co = neural_prophet('CO')
	mape_temp, predictions_temp = neural_prophet('TEMP')

	# Write to excel
	with pd.ExcelWriter('Predictions_NeuralProphet.xlsx') as writer:
		predictions_co.to_excel(writer, sheet_name = 'CO_NeuralProphet')
		predictions_temp.to_excel(writer, sheet_name = 'Temp_NeuralProphet')

	return mape_co, mape_temp, predictions_co, predictions_temp
