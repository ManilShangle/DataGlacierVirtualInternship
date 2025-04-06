import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify

application = Flask(__name__)

# Load model
import pickle
model = pickle.load(open("model.pkl", "rb"))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare data for model
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        # Map prediction to meaningful label
        class_mapping = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
        prediction_label = class_mapping.get(prediction, "Unknown")

        # Generate a visualization
        fig, ax = plt.subplots()
        ax.bar(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
               [sepal_length, sepal_width, petal_length, petal_width],
               color=['blue', 'green', 'red', 'purple'])
        ax.set_title("Feature Values")

        # Convert plot to image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', 
                               prediction_text=f'The predicted flower species is: {prediction_label}',
                               plot_url=plot_url)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    application.run(debug=True)
