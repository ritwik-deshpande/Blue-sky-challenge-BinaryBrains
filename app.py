from flask import Flask, request
app = Flask(__name__)
from render_model_op import *

model_op_mapper =dict()
model_op_mapper['arima'] = get_arima_op
model_op_mapper['fbprophet'] = get_fbprophet_op
model_op_mapper['ses'] = get_ses_op
model_op_mapper['neuralprophet'] = get_neuralprophet_op
model_op_mapper['lstm'] = get_lstm_op

models = ['lstm','arima', 'fbprophet', 'ses', 'neuralprophet']

@app.route('/')
def home():
    return render_template('line_chart.html', models = models, title='Temp and CO Forecasting', max=17000, labels=[],
                           actual_temp = [], predicted_temp = [], actual_co = [], predicted_co = [])


@app.route('/submit_model')
def get_model_op():
    model_name = request.args['model']
    model_index = models.index(model_name)
    models[model_index] = models[0]
    models[0] = model_name
    return model_op_mapper[model_name](models)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)