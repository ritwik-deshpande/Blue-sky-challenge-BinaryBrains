# Blue-sky-challenge-BinaryBrains

This application is a submission by the team BinaryBrains for [The Blue Sky Challenge](https://www.hackerearth.com/challenges/hackathon/ieee-machine-learning-hackathon/) under sub-theme 2: Blue Sky Below, Forecasting Sensor Measurements in Smart Air Quality Monitoring System

# Codebase details
1. The input data 'AirQuality.csv' is contained in Data folder.
2. baseline_models folder contains the implementation of 4 baseline models - Exponential Smoothing, Auto ARIMA, Facebook Prophet, Neural Prophet (.py files and google colab notebook)
3. The predictions are stored as excel files in the Data folder.
4. LSTM folder contains the LSTM model code and the trained models saved as hd5 files.
5. render_model_op.py renders the predictions to UI (hosted on cloud platform Heroku).

# Steps to view the web app 
This is a flask application to view the comparisons between the actual and predicted air-quality parameters(CO and Temp)

The application is deployed on Heroku cloud platform and can be accessed at [Temp and CO Forecasting](https://predict-air-quality-app.herokuapp.com/).
On the cloud platform there were certain hinderance to install all the required packages to run the models, so the results shown is using stored excel files generated using the trained models from the application.

To build & run the application in your local:

1. Clone [Blue-sky-challenge-BinaryBrains](https://github.com/ritwik-deshpande/Blue-sky-challenge-BinaryBrains.git)
2. Install the python packages given in the requirements.txt. Use the command `pip install -r requirements.txt`
3. Run py files directly from CLI interface, for example:
      1. LSTM temperature model -> python .\LSTM\py_files\LSTM_temp.py
      2. LSTM CO model -> python .\LSTM\py_files\LSTM_co.py
4. Access the application at your localhost on 8080 port

#### Note
To generate predicitons, run the google colab notebooks and save the excel files generated to the data folder. A deployment will be triggered on Heroku and the results can be viewed.
