import os
import cv2
import matplotlib.pyplot as plt
from src.data_loader import IDRiDDataset
import numpy as np

def verify_preprocessing():
    # Paths
    root_dir = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\1. Original Images\a. Training Set"
    csv_file = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
    
    # Initialize Dataset
    dataset = IDRiDDataset(root_dir=root_dir, csv_file=csv_file)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get 5 samples
    indices = [0, 10, 20, 30, 40]
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 20))
    
    for i, idx in enumerate(indices):
        # Get processed image from dataset
        processed_img_pil, label = dataset[idx]
        processed_img = np.array(processed_img_pil)
        
        # Get original image manually for comparison
        img_name = dataset.labels_df.iloc[idx]['Image name']
        img_path = os.path.join(root_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
             img_path = os.path.join(root_dir, img_name + ".tif")
        
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Plot Original
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original: {img_name} (Grade {label})")
        axes[i, 0].axis('off')
        
        # Plot Processed
        axes[i, 1].imshow(processed_img)
        axes[i, 1].set_title(f"Processed: {processed_img.shape}")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    output_path = r"c:\Users\deepak\.gemini\antigravity\brain\9962d4fd-e56f-4322-b8bb-d43ba543f9fd\preprocessing_check.png"
    plt.savefig(output_path)
    print(f"Verification image saved to: {output_path}")

if __name__ == "__main__":
    verify_preprocessing()
