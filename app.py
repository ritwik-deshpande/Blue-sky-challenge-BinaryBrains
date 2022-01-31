from flask import Flask, request
app = Flask(__name__)
from render_model_op import *

model_op_mapper =dict()
model_op_mapper['Auto_Arima'] = get_arima_op
model_op_mapper['FaceBook_Prophet'] = get_fbprophet_op
model_op_mapper['Exponential_Smoothing'] = get_ses_op
model_op_mapper['NeuralProphet'] = get_neuralprophet_op
model_op_mapper['LSTM'] = get_lstm_op

models = ['LSTM','Auto_Arima', 'FaceBook_Prophet', 'Exponential_Smoothing', 'NeuralProphet']

@app.route('/')
def home():
    return render_template('line_chart.html', models = models, title='Temp and CO Forecasting', max=17000, labels=[],
                           actual_temp = [], predicted_temp = [], actual_co = [], predicted_co = [])


@app.route('/submit_model')
def get_model_op():
    try:
        model_name = request.args['model']
        model_index = models.index(model_name)
        models[model_index] = models[0]
        models[0] = model_name
        return model_op_mapper[model_name](models)
    except Exception as ex:
        return str(ex)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)