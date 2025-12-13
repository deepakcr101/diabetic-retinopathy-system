import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import cv2
import os
from src.data_loader import IDRiDDataset
from src.model import HybridDRModel
from src.explainability import generate_diagnostic_report

def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    
    # Paths
    test_dir = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\1. Original Images\b. Testing Set"
    test_csv = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"
    model_path = "best_model.pth"
    output_dir = r"c:\Users\deepak\.gemini\antigravity\brain\9962d4fd-e56f-4322-b8bb-d43ba543f9fd\reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    test_dataset = IDRiDDataset(test_dir, test_csv, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load Model
    model = HybridDRModel(num_classes=5).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded best model weights.")
    else:
        print("Warning: best_model.pth not found. Using random weights.")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Quadratic Kappa: {kappa:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print("="*30)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))
    
    # Generate Diagnostic Reports for a few samples
    print("Generating Diagnostic Reports...")
    indices = [0, 10, 20, 30, 40] # Sample indices
    
    for idx in indices:
        if idx >= len(test_dataset): continue
        
        # Get data
        img_tensor, label = test_dataset[idx]
        
        # Get original image for visualization
        img_name = test_dataset.labels_df.iloc[idx]['Image name']
        img_path = os.path.join(test_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
             img_path = os.path.join(test_dir, img_name + ".tif")
             
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (512, 512))
        original_img = original_img / 255.0 # Normalize for overlay
        
        # Generate Report
        report_img, pred_idx, conf = generate_diagnostic_report(model, img_tensor, original_image=original_img, device=DEVICE)
        
        # Save
        save_path = os.path.join(output_dir, f"report_{img_name}_GT{label}_Pred{pred_idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR))
        print(f"Saved report: {save_path}")

if __name__ == "__main__":
    evaluate_model()
