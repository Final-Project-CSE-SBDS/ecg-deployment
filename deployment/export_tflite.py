import os
import tensorflow as tf

import sys

def convert_onnx_to_tflite(onnx_path="./deployment/mamba_ecg.onnx", tf_model_dir="./deployment/tf_model", tflite_path="./deployment/mamba_ecg.tflite"):
    print("Converting ONNX to TensorFlow SavedModel using onnx2tf...")
    
    # 1. Convert ONNX to TensorFlow
    # Forcing it to use the exact python environment's onnx2tf to avoid global PATH clashes
    ret = os.system(f'"{sys.executable}" -m onnx2tf -i {onnx_path} -o {tf_model_dir}')
    
    if ret != 0:
        print("[ERROR] Error during ONNX to TF conversion. Please strictly follow README instructions.")
        print("Ensure you have installed: pip install onnx2tf")
        return
        
    print(f"[SUCCESS] Successfully converted to TensorFlow SavedModel at {tf_model_dir}")
    
    print("Applying Quantization and converting to TFLite format for Wearable Devices...")
    
    # 2. Convert TF SavedModel to TFLite with FP16 Edge Optimization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    
    # Applying Float16 Quantization to reduce model size rapidly and speed up inference by 2x
    # while retaining nearly 100% of precision.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"[SUCCESS] Successfully exported edge-optimized Mamba TFLite model to '{tflite_path}'!")
    except Exception as e:
        print(f"[ERROR] TFLite Conversion Failed: {e}")

if __name__ == '__main__':
    convert_onnx_to_tflite()
