from flask import Flask, Markup, render_template

app = Flask(__name__)

labels = [
    'JAN', 'FEB', 'MAR', 'APR',
    'MAY', 'JUN', 'JUL', 'AUG',
    'SEP', 'OCT', 'NOV', 'DEC'
]

co_conc = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]

temperature = [
    96.67, 10.89, 1079.75, 139.19,
    22.91, 254.28, 2873.83, 764.87,
    49.29, 658.30, 9907, 697
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]


@app.route('/line')
def line():
    line_labels=labels
    return render_template('line_chart.html', title='Bitcoin Monthly Price in USD', max=17000, labels=line_labels,
                           co_conc = co_conc, temperature = temperature)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)