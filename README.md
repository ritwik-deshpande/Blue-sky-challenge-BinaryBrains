# Blue-sky-challenge-BinaryBrains
UI app for IEEE Hackathon
This is a flask application to view the comparisons between the actual and predicted air-quality parameters(CO and Temp)

The application is deployed on Heroku cloud platform and can be accessed at [Temp and CO Forecasting](https://predict-air-quality-app.herokuapp.com/)
At the cloud platform there were certain hinderance to install all the required packages to run the models, so the results shown is using stored excel files generated using the trained models from the application.

To build & run the application in your local:

1. Clone [Blue-sky-challenge-BinaryBrains](https://github.com/ritwik-deshpande/Blue-sky-challenge-BinaryBrains.git)
2. Install the python packages given in the requirements.txt
3. Run py files directly from CLI interface, for example:
      LSTM temperature model -> python3 LSTM_temp.py
      LSTM CO model -> python3 LSTM_co.py
4. Access the application at your localhost on 8080 port
