from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model

model = joblib.load('churn_model.pkl')


# Home Page

@app.route('/')
def home():
    return render_template('index.html')


# Prediction Input Page

@app.route('/predict')
def predict():
    return render_template('predict.html')


# Result Page

@app.route('/result', methods=['POST'])
def result():

    # Get form data

    tenure = int(request.form['tenure'])

    charges = float(request.form['charges'])

    senior = int(request.form['age'])

    # Create array for prediction

    data = np.array([[tenure, charges, senior]])

    # Predict churn

    prediction = model.predict(data)

    # Predict probability

    probability = model.predict_proba(data)

    # Convert prediction into text

    if prediction[0] == 1:

        result = "High Churn Risk"

    else:

        result = "Low Churn Risk"

    # Send data to HTML page

    return render_template(

        'result.html',

        prediction=result,

        probability=round(probability[0][1] * 100, 2),

        age=senior,

        charges=charges,

        tenure=tenure
    )


# Dashboard Page

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Run Flask App

if __name__ == '__main__':
    app.run(debug=True)

