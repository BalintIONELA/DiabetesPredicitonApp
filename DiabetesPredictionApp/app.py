import numpy as np
from flask import Flask, request, render_template
from waitress import serve
import pickle

app = Flask(__name__)

# Load model
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        # Retrieve and transform input data
        int_features = [x for x in request.form.values()]
        final_features = [np.array(int_features, dtype=float)]

        # Check for negative values
        if any(value < 0 for value in final_features[0]):
            raise ValueError("Negative values are not allowed.")

        # Debug: Show input values
        print("Input features:", final_features)

        # Make prediction
        prediction = model.predict(final_features)

        # Debug: Show prediction
        print("Prediction:", prediction)

        if prediction[0] == 0:
            output = 'You are not Diabetic'
        else:
            output = 'You are Diabetic'
    except ValueError as e:
        output = str(e)
        print("Error:", str(e))
    except Exception as e:
        output = f"An error occurred: {str(e)}"
        print("Error:", str(e))

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=50100, threads=2)
