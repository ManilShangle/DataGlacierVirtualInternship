# app.py
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # a simple HTML form for inputs

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form request
    data = request.form
    # Convert input strings to floats (assuming 4 input features for Iris)
    try:
        features = [float(data.get(f'feature{i}', 0)) for i in range(1, 5)]
    except Exception as e:
        return jsonify({"error": "Invalid input. Please provide numeric values."})
    
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
