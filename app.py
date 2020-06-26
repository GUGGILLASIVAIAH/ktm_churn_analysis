# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:28:55 2020

@author: GUGGILLA SIVAIAH
"""

import numpy as np
import pickle
from flask import Flask, render_template, jsonify, request

# initialising the flask application

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# routing the applicatin to root folder

@app.route('/')
def home():
    return render_template('index.html')

# rooting to the predictin outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    output = prediction[0][1]
    
    return render_template('index.html', prediction_text='Chance of customer purchae a bike{}'.format(round(output,2)))

if __name__=='__main__':
    app.run(debug=True)