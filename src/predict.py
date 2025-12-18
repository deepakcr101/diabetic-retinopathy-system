import torch
import cv2
import argparse
import os
import sys
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HybridDRModel
from src.explainability import generate_diagnostic_report

def predict_single_image(image_path, model_path="best_model_qwk.pth"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path}...")
    model = HybridDRModel(num_classes=5).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return

    # Load and Preprocess for Model
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read image.")
        return
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img_resized = cv2.resize(original_img, (512, 512))
    norm_img_for_overlay = original_img_resized / 255.0

    img_tensor = transform(original_img_resized)

    # Generate Report
    print("Running inference and generating explanation...")
    report_img, pred_idx, confidence = generate_diagnostic_report(
        model, img_tensor, original_image=norm_img_for_overlay, device=DEVICE
    )

    # DR Grading mapping
    grades = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
    prediction_str = grades.get(pred_idx, "Unknown")

    print("\n" + "="*30)
    print(f"PREDICTION: {prediction_str} (Grade {pred_idx})")
    print(f"CONFIDENCE: {confidence:.2f}")
    print("="*30 + "\n")

    # Save Result
    os.makedirs("results", exist_ok=True)
    save_path = "results/demo_prediction.png"
    cv2.imwrite(save_path, cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR))
    print(f"Diagnosis report saved to: {save_path}")
    
    # Try to open the image automatically (Windows)
    try:
        os.startfile(os.path.abspath(save_path))
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DR severity for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the retinal image")
    parser.add_argument("--model", type=str, default="best_model_qwk.pth", help="Path to trained model weights")
    
    args = parser.parse_args()
    predict_single_image(args.image, args.model)
