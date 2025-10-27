# ==========================================================
# 🧠 AI Powered Medical Image Analysis
# ----------------------------------------------------------
# Multi-class disease detection using Deep Learning (ResNet18)
# Detects: Normal, Pneumonia, COVID-19, and Tuberculosis
# Includes Grad-CAM for explainability and Gradio Web UI
# ==========================================================

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2


# -----------------------------
# ⚙️ Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 🩺 Disease Classes
# -----------------------------
classes = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

# -----------------------------
# 🧠 Load Pre-trained Multi-Class Model (ResNet18)
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model = model.to(device)
model.eval()


# -----------------------------
# 🧩 Image Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -----------------------------
# 🔍 Grad-CAM Hook Setup
# -----------------------------
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


# -----------------------------
# 🎯 Grad-CAM Generator
# -----------------------------
def generate_gradcam(img_tensor, class_idx):
    global activations, gradients
    activations, gradients = None, None

    # Forward
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    prob = probs[0, class_idx].item()

    # Backward
    model.zero_grad()
    output[0, class_idx].backward()

    # Compute Grad-CAM
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)

    # Resize and colorize
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap, prob


# -----------------------------
# 🔮 Prediction + Grad-CAM Overlay
# -----------------------------
def analyze_medical_image(image):
    try:
        img_resized = image.resize((224, 224))
        img_tensor = transform(img_resized.convert("RGB")).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]

        # Grad-CAM for top prediction
        heatmap, _ = generate_gradcam(img_tensor, pred_idx)
        img_np = np.array(img_resized)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Convert to dictionary
        results = {cls: float(probs[i]) for i, cls in enumerate(classes)}

        return results, overlay

    except Exception as e:
        return {"Error": str(e)}, None


# -----------------------------
# 🌐 Gradio Web Interface
# -----------------------------
interface = gr.Interface(
    fn=analyze_medical_image,
    inputs=gr.Image(type="pil", label="Upload X-ray or CT Image"),
    outputs=[
        gr.Label(num_top_classes=4, label="Predicted Disease Probabilities"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap Visualization")
    ],
    title="🧠 AI Powered Medical Image Analysis",
    description=(
        "Upload an X-ray or CT scan image to detect diseases such as Normal, "
        "Pneumonia, COVID-19, and Tuberculosis using a ResNet18 model. "
        "A Grad-CAM heatmap highlights regions most influential in the model's prediction."
    ),
    examples=[
        ["examples/normal.jpg"],
        ["examples/pneumonia.jpg"],
        ["examples/covid.jpg"],
        ["examples/tuberculosis.jpg"]
    ]
)


# -----------------------------
# 🚀 Launch App
# -----------------------------
if __name__ == "__main__":
    interface.launch(debug=True, share=True) 
