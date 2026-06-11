from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open('churn_model.pkl', 'rb'))
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    data = pd.read_csv(filepath)

    churn_count = 0
    no_churn_count = 0

    for index, row in data.iterrows():

        tenure = row['tenure']
        charges = row['MonthlyCharges']

        prediction = model.predict([[tenure, charges]])

        result = prediction[0]

        if result == 1:
            churn_count += 1
        else:
            no_churn_count += 1

    total = churn_count + no_churn_count

    churn_percent = (churn_count / total) * 100
    no_churn_percent = (no_churn_count / total) * 100

    output = f"""
    Total Customers: {total}
    Churn: {churn_percent:.2f}%
    No Churn: {no_churn_percent:.2f}%
    """

    return render_template('index.html', result=output)

if __name__ == '__main__':
    app.run(debug=True)

