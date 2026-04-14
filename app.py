import logging
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "saved_models/model.onnx"
session = None
input_name = None

def load_model():
    global session, input_name
    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Invalid request: missing "input" field in JSON body'}), 400

        input_data = data['input']
        if not isinstance(input_data, list):
            return jsonify({'error': '"input" must be a list of ECG values'}), 400

        seq_len = len(input_data)
        logger.info(f"Received predict request with input length: {seq_len}")

        # Ensure correct reshape for the model. 
        expected_shape = session.get_inputs()[0].shape
        
        # Resolve dynamic dimensions (None or string) to 1
        shape = [1 if not isinstance(dim, int) else dim for dim in expected_shape]
        
        tensor = np.array(input_data, dtype=np.float32)
        try:
            tensor = tensor.reshape(shape)
        except ValueError as e:
            # Fallback if the expected length doesn't perfectly match
            logger.warning(f"Shape mismatch: requested {shape} but got length {seq_len}. Expanding dims.")
            tensor = np.expand_dims(tensor, axis=0) # shape (1, seq_len)

        outputs = session.run(None, {input_name: tensor})
        
        # Handle outputs depending on shape
        output_data = np.array(outputs).flatten()
        
        prediction_idx = 0
        confidence = 0.0

        if len(output_data) >= 2:
            # Apply softmax for confidence
            exp_preds = np.exp(output_data - np.max(output_data))
            probs = exp_preds / exp_preds.sum()
            prediction_idx = int(np.argmax(probs))
            confidence = float(probs[prediction_idx])
        else:
            prob = float(output_data[0])
            
            # Make sure it's a sigmoid prob (between 0 and 1)
            if prob < 0 or prob > 1:
                prob = 1 / (1 + np.exp(-prob)) # Apply sigmoid if values look like logits
                
            prediction_idx = 1 if prob >= 0.5 else 0
            confidence = prob if prediction_idx == 1 else 1.0 - prob

        labels = ["Normal", "Arrhythmia"]
        label = labels[prediction_idx] if prediction_idx < len(labels) else "Unknown"

        logger.info(f"Prediction completed: {label} (Confidence: {confidence:.4f})")
        
        return jsonify({
            'prediction': label,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
