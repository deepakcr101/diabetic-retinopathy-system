import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook for gradients
        self.target_layer.register_backward_hook(self.save_gradient)
        
        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for specific class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().detach().numpy(), output

def overlay_heatmap(img, heatmap):
    """
    Overlays a heatmap on an image.
    img: (H, W, 3) numpy array, range [0, 255] or [0, 1]
    heatmap: (H, W) numpy array, range [0, 1]
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
        
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def generate_diagnostic_report(model, image_tensor, original_image, device):
    """
    Generates a diagnostic report with Grad-CAM.
    """
    model.eval()
    
    # Initialize Grad-CAM on the last convolutional layer of the backbone
    # For ResNet50, this is usually layer4[-1]
    target_layer = list(model.backbone.children())[-1][-1].conv3
    grad_cam = GradCAM(model, target_layer)
    
    # Generate Heatmap
    heatmap, output = grad_cam(image_tensor.unsqueeze(0).to(device))
    
    # Get Prediction
    probs = F.softmax(output, dim=1)
    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    
    # Overlay
    result_img = overlay_heatmap(original_image, heatmap)
    
    return result_img, pred_idx, confidence
