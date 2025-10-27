# ==========================================================
# main.py
# Core Model Logic for AI-Powered Medical Image Analysis
# ==========================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2

# ==========================================================
# Device Setup
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Load Model
# ==========================================================
# Change to match number of diseases (1 = binary classification)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ==========================================================
# Image Preprocessing
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# Grad-CAM Setup
# ==========================================================
final_conv_layer = model.layer4[1].conv2
activations = None
gradients = None

def save_activation(module, input, output):
    global activations
    activations = output.detach()

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

final_conv_layer.register_forward_hook(save_activation)
final_conv_layer.register_backward_hook(save_gradient)

# ==========================================================
# Grad-CAM Generation
# ==========================================================
def generate_gradcam(img_tensor):
    global activations, gradients
    activations, gradients = None, None

    # Forward pass
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()

    # Backward pass
    model.zero_grad()
    output.backward(torch.ones_like(output))

    # Grad-CAM Calculation
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)

    # Resize & colorize
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap, prob

# ==========================================================
# Prediction + Grad-CAM Overlay
# ==========================================================
def predict_with_heatmap(image):
    try:
        img_resized = image.resize((224, 224))
        img_tensor = transform(img_resized.convert("RGB")).unsqueeze(0).to(device)

        heatmap, prob = generate_gradcam(img_tensor)

        img_np = np.array(img_resized)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        return {"Disease Detected": prob, "Healthy": 1 - prob}, overlay

    except Exception as e:
        return {"Error": str(e)}, None
