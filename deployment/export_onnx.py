
import sys
import os
import torch
import onnx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mamba_model import MambaECGClassifier

def export_to_onnx(model_path='./saved_models/best_mamba.pth', onnx_path='./deployment/mamba_ecg.onnx'):
    print("Preparing to export model to ONNX...")
    model = MambaECGClassifier(num_classes=2)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing weights file {model_path}. Train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input with shape (Batch, Sequence, Channels)
    dummy_input = torch.randn(1, 187, 1)
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    print("Exporting Pure PyTorch Mamba Model to ONNX format...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=18, # Update opset to 18 to fix PyTorch 2.5 Dynamo exporter natively
        do_constant_folding=True,
        input_names=['input_signal'], 
        output_names=['class_logits']
        # Removed dynamic_axes as TFLite prefers static shapes and Dynamo enforces static tracing by default
    )
    
    print(f"✅ Model successfully exported to {onnx_path}")
    
    # Verify ONNX model integrity
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is mathematically well-formed and verified!")
    except Exception as e:
        print(f"❌ ONNX Checker Warning: {e}")

if __name__ == '__main__':
    export_to_onnx()
