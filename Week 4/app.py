import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, url_for
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained model (assumed saved as 'model.pkl')
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapping for iris classes with descriptive names and reference image filenames.
iris_mapping = {
    0: {'name': 'Iris-setosa', 'image': 'iris_setosa.jpg'},
    1: {'name': 'Iris-versicolor', 'image': 'iris_versicolor.jpg'},
    2: {'name': 'Iris-virginica', 'image': 'iris_virginica.jpg'}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Extract features; expecting numbers for each iris measurement.
        features = [float(data.get(f'feature{i}', 0)) for i in range(1, 5)]
    except Exception as e:
        return jsonify({'error': 'Invalid input. Please provide numeric values.'}), 400

    # Make prediction using the loaded model.
    prediction = model.predict([features])[0]
    iris_info = iris_mapping.get(prediction, {'name': 'Unknown', 'image': 'unknown.jpg'})

    # Create a simple matplotlib plot that displays the prediction text.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f'Prediction: {iris_info["name"]}', fontsize=20,
            ha='center', va='center')
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Encode the matplotlib plot image as a base64 string.
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Build URL for the static reference image.
    iris_image_url = url_for('static', filename=iris_info['image'])
    
    return jsonify({
        'prediction': prediction,
        'prediction_text': iris_info['name'],
        'plot_image': plot_base64,
        'iris_image_url': iris_image_url
    })

if __name__ == '__main__':
    app.run(debug=True)
