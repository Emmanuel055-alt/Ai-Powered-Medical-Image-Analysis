# 🧠 AI-Powered Medical Image Analysis

**Multi-disease detection from medical images (Chest X-ray / CT) using PyTorch (ResNet18) with Grad-CAM explainability, a FastAPI backend, and a Gradio demo.**  

This project uses deep learning to classify medical images into multiple disease categories (e.g., Normal, Pneumonia, COVID-19, Tuberculosis) and provides interpretable visualizations using Grad-CAM.

---

## 🚀 Features

- 🩺 **Multi-disease classification** using ResNet18 (transfer learning)
- 🔍 **Explainable AI** with Grad-CAM heatmaps  
- 🌐 **Interactive web app** built with Gradio  
- ⚙️ **FastAPI backend** for API-based predictions  
- 📊 **Evaluation tools** — accuracy, confusion matrix, classification report  
- ☁️ **Colab-compatible** and GPU ready  

---

## 📁 Project Structure
give it in readme formta # 🧠 AI Powered Medical Image Analysis for Disease Detection

An AI-driven system that analyzes medical images (like X-rays or CT scans) and detects diseases such as **Pneumonia, COVID-19, and Tuberculosis**.  
It uses **deep learning (ResNet18)**, **Grad-CAM explainability**, and a **Gradio web interface** for interactive and visual predictions.

> ⚠️ *Disclaimer*: This project is for research and educational use only. It is **not** a substitute for medical diagnosis.

---

## 🚀 Features

- 🩺 Detects multiple diseases (Normal, Pneumonia, COVID-19, Tuberculosis)
- 🧠 Powered by **ResNet18** deep learning model
- 🔍 Visual explainability using **Grad-CAM**
- 🌐 Web-based interface using **Gradio**
- 📊 Model performance evaluation (Accuracy, Precision, Recall, F1)
- ☁️ Google Colab compatible (GPU ready)

---

## 📂 Project Structure

AI_Medical_Image_Analysis/
│
├── dataset/
│ ├── train/
│ │ ├── Normal/
│ │ ├── Pneumonia/
│ │ ├── COVID-19/
│ │ └── Tuberculosis/
│ ├── val/
│ └── test/
│
├── models/
│ └── best_model.pth
│
├── train_model.py
├── evaluate_accuracy.py
├── ui_app.py
├── gradcam_utils.py
├── requirements.txt
└── README.md

yaml
Copy code

---
## ⚙️ Installation

### 🧩 Step 1. Clone this Repository
```bash
git clone https://github.com/yourusername/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

### 🧱 Step 2. Create a Virtual Environment

To keep dependencies organized and avoid version conflicts, create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate

### 📦 Step 3. Install Dependencies

After activating your virtual environment, install all required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
🧾 requirements.txt
torch
torchvision
gradio
fastapi
uvicorn
opencv-python
Pillow
numpy
scikit-learn
matplotlib
tqdm
pandas
seaborn

## 🧠 Dataset Preparation

Organize your dataset as follows:

```bash
dataset/
├── train/
│   ├── Normal/
│   ├── Pneumonia/
│   ├── COVID-19/
│   └── Tuberculosis/
├── val/
│   ├── ...
└── test/
    ├── ...
## 🏋️‍♂️ Model Training

To train your own model, run the following command:

```bash
python train_model.py





## 📊 Model Evaluation (Accuracy & Metrics)

To evaluate the model's performance on the test dataset, run:

```bash
python evaluate_accuracy.py
Accuracy: 99.5%
Precision: 0.99
Recall: 0.99
F1 Score: 0.995



## 🔍 Grad-CAM Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** provides **visual interpretability** by highlighting which regions of the medical image influenced the model’s prediction the most.

### ⚙️ How It Works
1. Extracts gradients from the final convolution layer  
2. Computes importance weights based on gradient flow  
3. Generates a color heatmap overlay on the original image  

🟥 **Red areas** on the heatmap indicate regions that contributed most strongly to the prediction.

> 💡 **Why it matters:**  
> Grad-CAM helps build trust and transparency in AI models by showing *why* a particular diagnosis was made.

---

## 💻 Gradio Web App

Launch the AI-powered medical image analysis app:


python ui_app.py

!python ui_app.py

Then open the Gradio URL (e.g., https://xxxx.gradio.live) to use the interface.

App Functionality:
Upload an X-ray or CT image

View predicted disease probabilities

See Grad-CAM heatmap visualization

Example Output:

less
Copy code
Normal: 3.2%
Pneumonia: 85.4%
COVID-19: 6.1%
Tuberculosis: 5.3%

Predicted Disease: Pneumonia
🧩 Example Output Visualization
Input Image	Grad-CAM Heatmap

🌐 FastAPI Backend (Optional)
For developers who want to use REST API:

Run:

uvicorn api_app:app --host 0.0.0.0 --port 8000
Request Example:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/image.jpg"
Response Example:

json

{
  "prediction": "Pneumonia",
  "confidence": 85.4,
  "all_probabilities": {
    "Normal": 3.2,
    "Pneumonia": 85.4,
    "COVID-19": 6.1,
    "Tuberculosis": 5.3
  }
}

## ⚙️ Framework Overview

| Component | Description |
|------------|-------------|
| **Framework** | PyTorch |
| **Explainability** | Grad-CAM |
| **Interface** | Gradio |

---

## 🧱 Technologies Used

- 🐍 **Python 3.12+**  
- 🔥 **PyTorch**  
- 🖼️ **Torchvision**  
- 🌐 **Gradio**  
- 🎥 **OpenCV**  
- 📊 **Scikit-learn**  
- ⚡ **FastAPI** *(optional for REST API support)*  

---

## 🧮 Example Gradio UI Screenshot

> 🖥️ Interactive web application showcasing disease prediction with Grad-CAM explainability.

*(You can add a screenshot here, for example:)*  
```markdown
![Gradio App Screenshot](assets/gradio_ui.png)









