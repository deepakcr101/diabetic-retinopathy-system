import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import IDRiDDataset
from src.model import HybridDRModel

# --- CONFIGURATION ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
PHASE_1_EPOCHS = 5   # Frozen backbone
PHASE_2_EPOCHS = 20  # Fine-tuning
PHASE_1_LR = 3e-4
PHASE_2_LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def compute_loss_weights(labels, device):
    """Compute inverse class frequency weights."""
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        # Stats
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * accumulation_steps * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Valid", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return epoch_loss, epoch_acc, epoch_qwk

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data & Stratified Split
    train_csv = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
    train_df_full = pd.read_csv(train_csv)
    train_df_full.columns = [c.strip() for c in train_df_full.columns]
    
    # Stratified Split (80/20)
    train_df, val_df = train_test_split(
        train_df_full, 
        test_size=0.2, 
        stratify=train_df_full['Retinopathy grade'], 
        random_state=SEED
    )
    
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    
    # 2. Datasets & Loaders
    root_dir = r"c:\Users\deepak\Projects\diabetic-retinopathy-system\datasets\B. Disease Grading\1. Original Images\a. Training Set"
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = IDRiDDataset(root_dir, train_df, transform=train_transforms)
    val_dataset = IDRiDDataset(root_dir, val_df, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Model & Loss
    model = HybridDRModel(num_classes=5).to(DEVICE)
    loss_weights = compute_loss_weights(train_df['Retinopathy grade'], DEVICE)
    print(f"Computed Class Weights: {loss_weights}")
    
    # Weighted Cross Entropy with Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    
    # --- PHASE 1: FROZEN BACKBONE ---
    print("\n=== PHASE 1: FROZEN BACKBONE TRAINING ===")
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE_1_LR)
    
    best_qwk = -1.0
    
    for epoch in range(PHASE_1_EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS)
        v_loss, v_acc, v_qwk = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{PHASE_1_EPOCHS} | T_Loss: {t_loss:.4f} | V_Loss: {v_loss:.4f} | V_Acc: {v_acc:.4f} | V_QWK: {v_qwk:.4f}")
        
        if v_qwk > best_qwk:
            best_qwk = v_qwk
            torch.save(model.state_dict(), "best_model_qwk.pth")
            print("  --> Saved Best Model (QWK improved)")

    # --- PHASE 2: FINE TUNING ---
    print("\n=== PHASE 2: FULL FINE-TUNING ===")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=PHASE_2_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Load best weights from Phase 1 to ensure we start from the peak
    if os.path.exists("best_model_qwk.pth"):
        model.load_state_dict(torch.load("best_model_qwk.pth", map_location=DEVICE))
        print("Loaded best weights from Phase 1")
    
    for epoch in range(PHASE_2_EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS)
        v_loss, v_acc, v_qwk = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{PHASE_2_EPOCHS} (FT) | T_Loss: {t_loss:.4f} | V_Loss: {v_loss:.4f} | V_Acc: {v_acc:.4f} | V_QWK: {v_qwk:.4f}")
        
        if v_qwk == 0.0 or v_qwk < 0:
            print("  [WARNING] Metric Collapse (QWK <= 0). Predictions might be degenerate.")
        
        # Scheduler steps on Validation QWK (metric we care about)
        scheduler.step(v_qwk)
        
        if v_qwk > best_qwk:
            best_qwk = v_qwk
            torch.save(model.state_dict(), "best_model_qwk.pth")
            print("  --> Saved Best Model (QWK improved)")

    print(f"\nTraining Complete. Global Best QWK: {best_qwk:.4f}")

if __name__ == "__main__":
    main()
