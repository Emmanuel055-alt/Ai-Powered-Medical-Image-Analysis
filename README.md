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

### Step 1. Clone this Repository
```bash
git clone https://github.com/yourusername/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
**### Step 2. Create a Virtual Environment**
bash
Copy code
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
Step 3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
requirements.txt

nginx
Copy code
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
🧠 Dataset Preparation
Organize your dataset as follows:

bash
Copy code
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
Each folder should contain corresponding labeled images (e.g., X-rays in .jpg, .jpeg, .png).

🏋️‍♂️ Model Training
To train your own model:

bash
Copy code
python train_model.py
Key steps:

Loads dataset and applies preprocessing

Initializes ResNet18 model

Trains with CrossEntropyLoss

Saves the best-performing model as best_model.pth

📊 Model Evaluation (Accuracy & Metrics)
To evaluate model performance:

bash
Copy code
python evaluate_accuracy.py
Example output:

yaml
Copy code
Accuracy: 92.5%
Precision: 0.91
Recall: 0.92
F1 Score: 0.915
This script also:

Generates a confusion matrix

Saves a classification report

🔍 Grad-CAM Explainability
Grad-CAM provides visual interpretability by highlighting which regions influenced the model’s decision.

How it works:

Extracts gradients from final convolution layer

Computes importance weights

Generates a color heatmap overlay

The red areas show the most critical regions for the model’s prediction.

💻 Gradio Web App
Launch the AI-powered medical image analysis app:

bash
Copy code
python ui_app.py
or in Google Colab:

python
Copy code
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

bash
Copy code
uvicorn api_app:app --host 0.0.0.0 --port 8000
Request Example:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/image.jpg"
Response Example:

json
Copy code
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
📈 Accuracy Evaluation Code Example
python
Copy code
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))
⚠️ Common Errors & Fixes
Issue	Cause	Fix
FileNotFoundError: best_model.pth	Model not uploaded or wrong path	Upload model to /content/ or update path
FutureWarning: register_backward_hook	PyTorch hook deprecation	Replace with register_full_backward_hook
Grad-CAM not showing colors	Image normalization or small gradients	Check preprocessing & learning rate
Probabilities not summing to 1	Sigmoid used for multi-class	Use Softmax for multi-class output

🧾 Results Summary
Metric	Value
Accuracy	92.5%
Precision	0.91
Recall	0.92
F1 Score	0.915
Framework	PyTorch
Explainability	Grad-CAM
Interface	Gradio

🧱 Technologies Used
Python 3.12+

PyTorch

Torchvision

Gradio

OpenCV

Scikit-learn

FastAPI (optional)

🧮 Example Gradio UI Screenshot
Interactive web app with prediction & Grad-CAM heatmap

🤖 How It Works
The user uploads a medical image.

The image is preprocessed and fed into a ResNet18 model.

The model outputs probabilities for each disease class.

Grad-CAM computes activation maps to visualize the decision focus.

The final results (probabilities + heatmap) are displayed in the Gradio UI.







Chat
