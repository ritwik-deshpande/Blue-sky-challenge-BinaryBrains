import pandas as pd
from flask import render_template

days = ['2004-03-18','2004-03-19','2004-03-20','2004-03-21','2004-03-22','2004-03-23','2004-03-24']

def calculate_mape(actual, forecast):
    APE = []
    for day in range(len(actual)):
        # Calculate percentage error
        per_err = (actual[day] - forecast[day]) / actual[day]

        # Take absolute value of
        # the percentage error (APE)
        per_err = abs(per_err)

        # Append it to the APE list
        APE.append(per_err)

    # Calculate the MAPE
    MAPE = sum(APE) / len(APE)
    return round(MAPE, 3)


def get_mape_values(actual, forecast):
    MAPE = []
    for day in range(0, 7):
        MAPE.append(calculate_mape(actual[day*24 : (day + 1)*24], forecast[day*24 : (day + 1)*24] ))

    return MAPE

""" Temp & CO predictions using ARIMA """
def get_arima_op(models):
    arima_co_op = pd.read_excel('./Data/Predictions_AutoArima.xlsx',  sheet_name='CO_AutoArima')
    arima_temp_op = pd.read_excel('./Data/Predictions_AutoArima.xlsx', sheet_name='Temp_AutoArima')

    mape_values_temp = get_mape_values(arima_temp_op['TEMP'].to_list(), arima_temp_op['PREDICTED_TEMP_ARIMA'].to_list())
    mape_values_co = get_mape_values(arima_co_op['CO'].to_list(), arima_co_op['PREDICTED_CO_ARIMA'].to_list())

    return render_template('line_chart.html',
                           days = days,
                           avg_mape_co= round(sum(mape_values_co) / len(mape_values_co),3),
                           avg_mape_temp=round(sum(mape_values_temp) / len(mape_values_temp),3),
                           mape_values_temp = mape_values_temp,
                           mape_values_co = mape_values_co,
                           models = models, model_name = 'Arima', title = 'Temp and CO Predictions by ARIMA', max=100,
                           labels= arima_co_op['DATE_TIME'].to_list(),
                           actual_temp = arima_temp_op['TEMP'].to_list(),
                           predicted_temp = arima_temp_op['PREDICTED_TEMP_ARIMA'].to_list(),
                           actual_co = arima_co_op['CO'].to_list(),
                           predicted_co = arima_co_op['PREDICTED_CO_ARIMA'].to_list())

""" Temp & CO predictions using SES """
def get_ses_op(models):
    ses_co_op = pd.read_excel('./Data/Predictions_ExponentialSmoothing.xlsx', sheet_name='CO_ExponentialSmoothing')
    ses_temp_op = pd.read_excel('./Data/Predictions_ExponentialSmoothing.xlsx', sheet_name='Temp_ExponentialSmoothing')

    mape_values_temp = get_mape_values(ses_temp_op['TEMP'].to_list(), ses_temp_op['PREDICTED_TEMP_SES'].to_list())
    mape_values_co = get_mape_values(ses_co_op['CO'].to_list(), ses_co_op['PREDICTED_CO_SES'].to_list())



    return render_template('line_chart.html',
                           days = days,
                           avg_mape_co=round(sum(mape_values_co) / len(mape_values_co),3),
                           avg_mape_temp=round(sum(mape_values_temp) / len(mape_values_temp),3),
                           mape_values_temp = mape_values_temp,
                           mape_values_co = mape_values_co,
                           models = models,model_name = 'Exponential Smoothening',
                           title= 'TEMP Predictions by SES', max=100,
                           labels= ses_co_op['DATE_TIME'].to_list(),
                            actual_temp = ses_temp_op['TEMP'].to_list(),
                           predicted_temp = ses_temp_op['PREDICTED_TEMP_SES'].to_list(),
                            actual_co = ses_co_op['CO'].to_list(),
                           predicted_co = ses_co_op['PREDICTED_CO_SES'].to_list())



""" Temp & CO predictions using Fb Prophet """
def get_fbprophet_op(models):
    fbprophet_co_op = pd.read_excel('./Data/Predictions_FbProphet.xlsx', sheet_name='CO_FbProphet')
    fbprophet_temp_op = pd.read_excel('./Data/Predictions_FbProphet.xlsx', sheet_name='Temp_FbProphet')

    mape_values_temp = get_mape_values(fbprophet_temp_op['TEMP'].to_list(), fbprophet_temp_op['PREDICTED_TEMP_FBPROPHET'].to_list())
    mape_values_co = get_mape_values(fbprophet_co_op['CO'].to_list(), fbprophet_co_op['PREDICTED_BFILL_CO_FBPROPHET'].to_list())

    return render_template('line_chart.html',
                           days = days,
                           avg_mape_co=round(sum(mape_values_co) / len(mape_values_co),3),
                           avg_mape_temp=round(sum(mape_values_temp) / len(mape_values_temp),3),
                           mape_values_temp=mape_values_temp,
                           mape_values_co=mape_values_co,
                           models = models,model_name = 'FB Prophet' ,title= 'TEMP Predictions by Fb Prophet', max=100,
                           labels= fbprophet_co_op['DATE_TIME'].to_list(),
                            actual_temp = fbprophet_temp_op['TEMP'].to_list(),
                           predicted_temp = fbprophet_temp_op['PREDICTED_TEMP_FBPROPHET'].to_list(),
                            actual_co = fbprophet_co_op['CO'].to_list(),
                           predicted_co = fbprophet_co_op['PREDICTED_BFILL_CO_FBPROPHET'].to_list())


"""" Temp & CO predictions using Neural Prophet """
def get_neuralprophet_op(models):
    neuralprophet_co_op = pd.read_excel('./Data/Predictions_NeuralProphet.xlsx', sheet_name='CO_NeuralProphet')
    neuralprophet_temp_op = pd.read_excel('./Data/Predictions_NeuralProphet.xlsx', sheet_name='Temp_NeuralProphet')

    mape_values_temp = get_mape_values(neuralprophet_temp_op['TEMP'].to_list(),
                                       neuralprophet_temp_op['PREDICTED_TEMP_NEURALPROPHET'].to_list())
    mape_values_co = get_mape_values(neuralprophet_co_op['CO'].to_list(), neuralprophet_co_op['PREDICTED_CO_NEURALPROPHET'].to_list())

    return render_template('line_chart.html',
                           days=days,
                           avg_mape_co=round(sum(mape_values_co) / len(mape_values_co),3),
                           avg_mape_temp=round(sum(mape_values_temp) / len(mape_values_temp),3),
                           mape_values_temp=mape_values_temp,
                           mape_values_co=mape_values_co,
                           models = models,model_name = 'Neural Prophet' ,title= 'TEMP Predictions by Neural Prophet', max=100,
                           labels= neuralprophet_co_op['DATE_TIME'].to_list(),
                            actual_temp = neuralprophet_temp_op['TEMP'].to_list(),
                           predicted_temp = neuralprophet_temp_op['PREDICTED_TEMP_NEURALPROPHET'].to_list(),
                            actual_co = neuralprophet_co_op['CO'].to_list(),
                           predicted_co = neuralprophet_co_op['PREDICTED_CO_NEURALPROPHET'].to_list())


"""" Temp & CO predictions using Neural Prophet """
def get_lstm_op(models):
    lstm_co_op = pd.read_csv('./Data/Predictions_LSTM_CO.csv')
    lstm_temp_op = pd.read_csv('./Data/Predictions_LSTM_Temp.csv')


    mape_values_temp = get_mape_values(lstm_temp_op['temp_actual'].to_list(),
                                       lstm_temp_op['temp_predict'].to_list())
    mape_values_co = get_mape_values(lstm_co_op['CO_actual'].to_list(), lstm_co_op['CO_predict'].to_list())

    return render_template('line_chart.html',
                           days=days,
                           avg_mape_co = round(sum(mape_values_co)/len(mape_values_co),3),
                           avg_mape_temp =round(sum(mape_values_temp) / len(mape_values_temp),3),
                           mape_values_temp=mape_values_temp,
                           mape_values_co=mape_values_co,
                           models = models,model_name = 'LSTM' ,title= 'TEMP Predictions by LSTM', max=100,
                           labels= lstm_co_op['Date'].to_list(),
                            actual_temp = lstm_temp_op['temp_actual'].to_list(),
                           predicted_temp = lstm_temp_op['temp_predict'].to_list(),
                            actual_co = lstm_co_op['CO_actual'].to_list(),
                           predicted_co = lstm_co_op['CO_predict'].to_list())
