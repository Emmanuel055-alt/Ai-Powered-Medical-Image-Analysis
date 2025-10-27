# ==========================================================
# app.py
# Web Interface for AI-Powered Medical Image Analysis
# ==========================================================

import gradio as gr
from main import predict_with_heatmap

# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_with_heatmap,
    inputs=gr.Image(type="pil", label="Upload Medical Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Results"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap Visualization")
    ],
    title="🧠 AI-Powered Medical Image Analysis for Disease Detection",
    description=(
        "Upload a medical image (X-ray, CT scan, MRI). "
        "The model predicts possible disease presence and displays "
        "Grad-CAM heatmaps highlighting critical regions of interest."
    ),
    allow_flagging=False
)

# ==========================================================
# Run the App
# ==========================================================
if __name__ == "__main__":
    interface.launch(debug=True, share=True)
