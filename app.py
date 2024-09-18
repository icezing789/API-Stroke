from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("stroke.pkl")

@app.route('/api/stroke', methods=['POST'])
def stroke():
    Gender = int(request.form.get('Gender')) 
    hypertension = int(request.form.get('hypertension')) 
    heartdisease = int(request.form.get('heartdisease')) 
    smoke = int(request.form.get('smoke')) 
    bmi = float(request.form.get('bmi')) 

    # Prepare the input for the model
    x = np.array([[Gender, hypertension, heartdisease, smoke, bmi]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    if int(prediction[0]) == 0:
        return {'stroke': 'ไม่เป็น'}, 200
    else:
        return {'stroke': 'เป็น'}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)