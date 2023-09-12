from flask import Flask,render_template,request

import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('lr_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    all_features = [np.array(features)]
    predicted = model.predict(all_features)

    return render_template('result.html', prediction='Car Price is {}'.format(predicted))


if __name__ == '__main__':
    app.run(debug=True)

