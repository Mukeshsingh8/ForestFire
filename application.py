import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

# Import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Capture input values from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        # Create an input array for prediction
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # Standardize the input data
        scaled_data = standard_scaler.transform(input_data)
        
        # Perform prediction using the ridge regression model
        prediction = ridge_model.predict(scaled_data)
        
        # Convert prediction to a readable format
        result = round(prediction[0], 2)  # Assuming prediction is a single float value
        
        # Render the home page with the prediction result
        return render_template('home.html', result=result)
    else:
        return render_template('home.html', result="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
