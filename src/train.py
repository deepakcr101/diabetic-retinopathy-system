import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.data_loader import IDRiDDataset
from src.model import HybridDRModel

def train_model():
    # --- CONFIGURATION FOR LARGE SCALE TRAINING ---
    BATCH_SIZE = 4            # Physical batch size (keep small for GPU memory)
    ACCUMULATION_STEPS = 8    # Virtural batch size = 4 * 8 = 32
    NUM_EPOCHS = 20           # Increased for real training
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    print(f"Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    # Paths
    train_dir = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\1. Original Images\a. Training Set"
    train_csv = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
    
    test_dir = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\1. Original Images\b. Testing Set"
    test_csv = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"
    
    # Transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = IDRiDDataset(train_dir, train_csv, transform=data_transforms)
    test_dataset = IDRiDDataset(test_dir, test_csv, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = HybridDRModel(num_classes=5).to(DEVICE)
    
    # Loss and Optimizer
    weight = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0]).to(DEVICE) # Optional: Class weighting for imbalance
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad() # Initialize zero gradients
        
        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Divide loss by accumulation steps
            loss = loss / ACCUMULATION_STEPS
            
            # Backward
            loss.backward()
            
            # Step Optimizer only after accumulating gradients
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Stats (multiply loss back for reporting)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * ACCUMULATION_STEPS * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(test_dataset)
        val_acc = val_corrects.double() / len(test_dataset)
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model.")
            
    print(f"Training Complete. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()
