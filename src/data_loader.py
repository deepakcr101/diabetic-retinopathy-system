import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
from src.preprocess_utils import crop_image_from_gray, apply_clahe

class IDRiDDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, phase='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            phase (string): 'train' or 'test'.
        """
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform
        self.phase = phase
        
        # Ensure 'Image name' column exists (handle potential whitespace)
        self.labels_df.columns = [c.strip() for c in self.labels_df.columns]
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.labels_df.iloc[idx]['Image name']
        # Handle extension differences if any, usually .jpg or .tif for IDRiD
        # Checking for file existence
        img_path = os.path.join(self.root_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
             img_path = os.path.join(self.root_dir, img_name + ".tif")

        # Read Image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing Steps
        # 1. Crop
        image = crop_image_from_gray(image)
        
        # 2. Resize (handled here or in transforms, but usually better here for consistency before CLAHE)
        image = cv2.resize(image, (512, 512))
        
        # 3. CLAHE
        image = apply_clahe(image)

        # Convert to PIL for Torchvision Transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            
        # Get Label
        # 'Retinopathy grade' is the target
        label = int(self.labels_df.iloc[idx]['Retinopathy grade'])

        return image, label
