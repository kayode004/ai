---
title: EMNIST Character Classifier
emoji: "🔤"
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "0.103.0"
app_file: app.py
pinned: false
---

# EMNIST Character Classifier 🔤

This Space predicts handwritten characters using two deep learning models trained on the EMNIST dataset.

## 🧠 Models Used:
- **emnist_letters_model.keras** – recognizes 26 capital letters (A–Z)
- **emnist_byclass_model.keras** – recognizes 62 characters (digits, uppercase, lowercase)

## 📂 How It Works
- **Backend**: FastAPI
- **Inference**: TensorFlow (.keras models)
- **Input**: User uploads a 28×28 grayscale image
- **Output**: Predicted character label

## 📦 Files
- `app.py`: FastAPI backend for inference
- `requirements.txt`: Python dependencies
- `emnist_letters_model.keras`: model trained to classify A–Z
- `emnist_byclass_model.keras`: model trained to classify 0–9, A–Z, a–z
