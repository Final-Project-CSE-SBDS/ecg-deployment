import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import sys
import argparse

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

CLASSES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

def run_wearable_inference(tflite_model_path='./deployment/mamba_ecg.tflite', input_csv=None):
    """
    Simulates real-time inference on a hardware edge device (like a smartwatch).
    """
    print("\n=====================================")
    print("⌚ Wearable ECG Monitor (Mamba AI)")
    print("=====================================")
    
    if not os.path.exists(tflite_model_path):
        print(f"[ERROR] TFLite model not found at '{tflite_model_path}'")
        print("Please run 'python main.py' to train and export the model first.")
        return

    # Initialize TF Lite Interpreter directly from the Edge
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if input_csv and os.path.exists(input_csv):
        print("Input Signal Loaded... (from CSV)")
        df = pd.read_csv(input_csv, header=None)
        signal = df.iloc[0].values[:187]
        if len(signal) < 187:
            signal = np.pad(signal, (0, 187 - len(signal)))
        dummy_signal = signal.reshape(1, 1, 187).astype(np.float32)
    else:
        print("Input Signal Loaded... ")
        # Generate an abnormal-looking sequence random pattern
        # This will simulate picking up an arrhythmia
        dummy_signal = np.random.rand(1, 1, 187).astype(np.float32) * 2.5
    
    print("Running Inference...")
    time.sleep(1.5) # Real-time simulation delay
    
    # Load into interpreter
    interpreter.set_tensor(input_details[0]['index'], dummy_signal)
    
    # Execute Linear-time inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_idx = np.argmax(output_data)
    
    # Calculate Softmax for better percentage probability visualization
    exp_out = np.exp(output_data - np.max(output_data))
    softmax = exp_out / np.sum(exp_out)
    confidence = softmax[predicted_idx] * 100
    
    cond_name = CLASSES[predicted_idx]
    
    if predicted_idx == 0:
        pred_text = f"NORMAL ✅"
    else:
        pred_text = f"{cond_name.upper()} ARRHYTHMIA ⚠️"
        
    print(f"\nPrediction: {pred_text}")
    print(f"Confidence: {confidence:.1f}%")
    print("-----------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wearable ECG Arrhythmia Monitor Demo")
    parser.add_argument('--input', type=str, help='Path to input ECG CSV file', default=None)
    args = parser.parse_args()
    
    run_wearable_inference(input_csv=args.input)
