from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature names must match those used during training
FEATURE_NAMES = [
    'alcohol',
    'flavanoids',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

# Mapping model output (0, 1, 2) to Cultivar names
CLASS_MAPPING = {
    0: "Cultivar 1",
    1: "Cultivar 2",
    2: "Cultivar 3"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json()
        
        # Ensure all required features are present
        input_data = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                input_data.append(float(data[feature]))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}'}), 400

        # Create DataFrame for prediction (preserves feature names for pipeline)
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Prediction
        prediction_idx = model.predict(input_df)[0]
        prediction_label = CLASS_MAPPING.get(prediction_idx, "Unknown")
        
        # Confidence Score (Max Probability)
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))

        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
