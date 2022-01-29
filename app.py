import pandas as pd
from turtle import st
from flask import Flask, request, Markup, render_template
from ml_models import auto_arima, fbprophet, ses
from ml_models import data_preprocess as dp
app = Flask(__name__)

# labels = [
#     'JAN', 'FEB', 'MAR', 'APR',
#     'MAY', 'JUN', 'JUL', 'AUG',
#     'SEP', 'OCT', 'NOV', 'DEC'
# ]

# co_conc = [
#     967.67, 1190.89, 1079.75, 1349.19,
#     2328.91, 2504.28, 2873.83, 4764.87,
#     4349.29, 6458.30, 9907, 16297
# ]

# temperature = [
#     96.67, 10.89, 1079.75, 139.19,
#     22.91, 254.28, 2873.83, 764.87,
#     49.29, 658.30, 9907, 697
# ]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

@app.route('/submit_model')
def line():
    text = request.form['model']
    return render_template('line_chart.html', title='Temp and CO Forecasting', max=17000, labels=[],
                           actual_temp = [], predicted_temp = [], actual_co = [], predicted_co = [])


""" Temp & CO predictions using ARIMA """
@app.route('/arima')
def arima_T():
    mape_arima, predictions_arima = auto_arima.arima_predictions_all()
    return render_template('line_chart.html', title= 'TEMP Predictions by ARIMA', max=100, labels= predictions_arima.index.to_list(),
                            actual_temp = predictions_arima['TEMP'].to_list(), predicted_temp = predictions_arima['PREDICTED_TEMP_ARIMA'].to_list(),
                            actual_co = predictions_arima['CO'].to_list(), predicted_co = predictions_arima['PREDICTED_CO_ARIMA'].to_list())


""" Temp & CO predictions using FB Prophet """
@app.route('/fbprophet')
def fb_prophet():
    mape_prophet, predictions_co, predictions_temp = fbprophet.fbprophet_predictions_all()
    return render_template('line_chart.html', title= 'CO Predictions by ARIMA', max=100, labels= predictions_co.index.to_list(),
                            actual_temp = predictions_temp['TEMP'].to_list(), predicted_temp = predictions_temp['PREDICTED_TEMP_PROPHET'].to_list(),
                            actual_co = predictions_co['CO'].to_list(), predicted_co = predictions_co['PREDICTED_CO_PROPHET'].to_list())


""" Temp & CO predictions using SES """
@app.route('/ses')
def ses_T():
    mape_ses, predictions_ses = ses.ses_predictions_all()
    print("SES_TEMP")
    return render_template('line_chart.html', title= 'TEMP Predictions by SES', max=100, labels= predictions_ses.index.to_list(),
                            actual_temp = predictions_ses['TEMP'].to_list(), predicted_temp = predictions_ses['PREDICTED_TEMP_SES'].to_list(),
                            actual_co = predictions_ses['CO'].to_list(), predicted_co = predictions_ses['PREDICTED_CO_SES'].to_list())


""" Temp & CO predictions using LSTM """
@app.route('/lstm')
def lstm_T():
    lstm_temp_pred = pd.read_csv('Data/temp_pred.csv', header=0, names=['Temp'])
    temp_pred_list = lstm_temp_pred['Temp'].to_list()
    mape_ses, predictions_ses = ses.ses_predictions_all()
    print("SES_TEMP")
    return render_template('line_chart.html', title= 'TEMP Predictions by SES', max=100, labels= predictions_ses.index.to_list(),
                            actual_temp = predictions_ses['TEMP'].to_list(), predicted_temp = temp_pred_list,
                            actual_co = predictions_ses['CO'].to_list(), predicted_co = predictions_ses['PREDICTED_CO_SES'].to_list())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)