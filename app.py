import pickle
import pandas as pd
import zipfile
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model from a pickle file
with zipfile.ZipFile('rainfall_model.zip', 'r') as zip_ref:
    zip_ref.extractall()  # Extract the content of the ZIP file
    # Assuming the extracted file is named 'rainfall_model.pkl'
    with open('rainfall_model.pkl', 'rb') as file:
        model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the inputs from the form
        inputs = {
            'MaxT': request.form.get('max_temp'),
            'MinT': request.form.get('min_temp'),
            'RH1': request.form.get('rh1'),
            'RH2': request.form.get('rh2'),
            'Wind': request.form.get('wind'),
            'SSH': request.form.get('ssh'),
            'Evap': request.form.get('evap'),
            'Radiation': request.form.get('radiation'),
            'FAO56_ET': request.form.get('fao56_et'),
            'Lat': request.form.get('lat'),
            'Lon': request.form.get('lon'),
            'Cum_Rain': request.form.get('cum_rain')
        }

        # Validate and preprocess the inputs
        input_data = []
        for key, value in inputs.items():
            if value is None or value.strip() == "":
                return render_template('error.html', error=f"Missing value for {key}")
            try:
                input_data.append(float(value))
            except ValueError:
                return render_template('error.html', error=f"Invalid input for {key}")

        # Make a prediction
        prediction = model.predict([input_data])[0]  # Assuming the model returns a single value

        return render_template('result.html', prediction=f"The predicted rainfall is: {prediction:.2f}")

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

