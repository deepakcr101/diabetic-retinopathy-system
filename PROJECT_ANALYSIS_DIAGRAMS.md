# Project Analysis and Data Flow Diagrams

## 1. Project Analysis

### **System Overview**
The **Diabetic Retinopathy Diagnostic System** is a hybrid deep learning framework designed to classify retinal fundus images into five grades of diabetic retinopathy (0-4). It integrates a **ResNet50 backbone** for spatial feature extraction with a **Transformer Encoder** for capturing global context, enhancing classification accuracy. Additionally, it includes an **Explainable AI (XAI)** module using **Grad-CAM** to provide visual diagnostic reports (heatmaps) aiding clinicians in understanding the model's decisions.

### **Architecture Components**
*   **Data Source**: IDRiD Dataset (Images + CSV Labels).
*   **Preprocessing Module**:
    *   **Cropping**: Removes uninformative black borders from fundus images using `crop_image_from_gray`.
    *   **Resizing**: Standardizes inputs to 512x512 pixels.
    *   **Enhancement**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast.
    *   **Normalization**: Standard ImageNet normalization.
*   **Model Architecture (`HybridDRModel`)**:
    *   **Feature Extractor**: ResNet50 (pre-trained, last layers removed).
    *   **Projection**: 1x1 Conv to reduce channels (2048 -> 512).
    *   **Global Context**: Transformer Encoder (2 layers, 8 heads).
    *   **Classifier**: Global Average Pooling + Fully Connected Layer.
*   **Training Pipeline**:
    *   Uses `CrossEntropyLoss` with class weighting to handle imbalance.
    *   `AdamW` optimizer.
    *   Gradient Accumulation for effective batch size management.
    *   Saves `best_model.pth` based on validation accuracy.
*   **Explainability (XAI)**:
    *   **Grad-CAM**: Computes gradients of the target class w.r.t. the last convolutional layer activations to generate attention heatmaps.

---

## 2. Data Flow Diagrams (DFD)

### **Context Diagram (Level 0 DFD)**
This diagram represents the system as a single high-level process interacting with external entities.

```mermaid
graph LR
    User[Common User / Ophthalmologist] -- "Provides Retinal Images" --> System((DR Diagnostic System))
    Dataset[IDRiD Dataset] -- "Training Images & Labels" --> System

    System -- "Diagnostic Report (Grade)" --> User
    System -- "Visual Explanation (Heatmap)" --> User
    System -- "Performance Metrics" --> Admin[System Admin/Developer]
```

### **Level 1 DFD (Logical Data Movements)**
This diagram breaks down the system into its major sub-processes and data stores.

```mermaid
graph TD
    %% External Entities
    InputData[IDRiD Dataset]
    User[User]

    %% Processes
    P1(1.0 Data Preprocessing)
    P2(2.0 Model Training)
    P3(3.0 Evaluation)
    P4(4.0 Diagnostic Generation XAI)

    %% Data Stores
    DS1[(Processed Tensor Cache)]
    DS2[(Model Weights .pth)]
    DS3[(Results & Logs)]

    %% Flows
    InputData -- "Raw Images + CSV" --> P1
    User -- "Input Image" --> P1
    
    P1 -- "Cleaned, Resized & Normalized Tensors" --> DS1
    
    DS1 -- "Batched Training Data" --> P2
    P2 -- "Forward Pass / Loss Calc" --> P2
    P2 -- "Update Weights" --> DS2
    
    DS1 -- "Validation Data" --> P3
    DS2 -- "Best Model Weights" --> P3
    P3 -- "Accuracy / Loss Metrics" --> DS3
    
    User -- "Request Explanation" --> P4
    DS2 -- "Loaded Model" --> P4
    P4 -- "Grad-CAM Heatmap + Prediction" --> User
```

---

## 3. System Flowchart

This flowchart illustrates the control logic from start to finish, including the training loop and inference logic.

```mermaid
flowchart TD
    Start([Start]) --> Mode{Select Mode}
    
    %% Training Branch
    Mode -- Training --> LoadConfig[Load Config: Hyperparams, Paths]
    LoadConfig --> InitDS[Initialize IDRiDDataset]
    InitDS --> Transform[Apply Preprocessing: Crop, Resize, CLAHE, Normalize]
    Transform --> DataLoader[Create DataLoaders Train/Test]
    DataLoader --> InitModel[Initialize HybridDRModel ResNet+Transformer]
    
    subgraph TrainingLoop [Training Loop Epoch 1..N]
        IterateBatch[Get Batch Images/Labels]
        Forward[Forward Pass]
        CalcLoss[Calculate CrossEntropy Loss]
        Backprop[Backpropagation]
        Optim[Optimizer Step w/ Grad Accumulation]
        CalcLoss -->|Accumulate Grads| Optim
    end
    
    InitModel --> IterateBatch
    Optim --> Validate[Validation Step]
    Validate --> CheckBest{Is Best Accuracy?}
    CheckBest -- Yes --> Save[Save best_model.pth]
    CheckBest -- No --> NextEpoch
    Save --> NextEpoch{More Epochs?}
    NextEpoch -- Yes --> IterateBatch
    NextEpoch -- No --> EndTraining([End Training])

    %% Inference / XAI Branch
    Mode -- Inference/XAI --> LoadWeights[Load best_model.pth]
    LoadWeights --> InputImg[Input Single Image]
    InputImg --> PreprocInf[Preprocess Image]
    PreprocInf --> Predict[Forward Pass -> Pred Class]
    Predict --> GradCAM[Compute Grad-CAM Gradients]
    GradCAM --> Heatmap[Generate Heatmap]
    Heatmap --> Overlay[Overlay on Original Image]
    Overlay --> Output([Show Report: Grade + Heatmap])

```
