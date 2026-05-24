# 🏥💡 Insurance Cost Prediction App using Deep Learning

A deep learning web application that predicts medical insurance charges based on patient details using a Neural Network built with TensorFlow/Keras.

---

## 🚀 Live Demo
Enter patient details → Get instant insurance cost prediction

---

## 📖 Project Overview

This project is an end-to-end Machine Learning + Deep Learning pipeline that:

- Loads and explores the insurance dataset  
- Performs data preprocessing and encoding  
- Scales features using StandardScaler  
- Trains a Deep Neural Network (DNN) using TensorFlow/Keras  
- Saves trained model and scaler for deployment  
- Deploys model using Streamlit / Gradio web app  

---

## 📁 Project Structure

insurance-cost-prediction-app/
│
├── insurance.csv
├── model.py
├── deployment.py
├── insurance_model.h5
├── scaler.pkl
├── requirements.txt
└── README.md

---

## 📊 Dataset Information

Dataset: Medical Cost Personal Dataset (Kaggle)

Features:
- age → Age of the person  
- sex → Gender  
- bmi → Body Mass Index  
- children → Number of dependents  
- smoker → Smoking status  
- region → Residential area  

Target:
- charges → Medical insurance cost  

---

## 🧠 Model Architecture

Input Layer  
↓  
Dense (128, ReLU)  
↓  
Dense (64, ReLU)  
↓  
Dense (32, ReLU)  
↓  
Output Layer (1, Linear)

---

## ⚙️ Model Configuration

- Optimizer: Adam  
- Loss: Mean Squared Error  
- Epochs: 100  
- Batch Size: 32  

---

## 🧹 Preprocessing Steps

- One-Hot Encoding for categorical features  
- Dropping unnecessary columns  
- Standard scaling using StandardScaler  
- Train-test split  

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Streamlit  


