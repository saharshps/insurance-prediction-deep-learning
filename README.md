🏥 Insurance Cost Prediction App using Deep Learning
A deep learning web application that predicts medical insurance charges based on patient details using a Neural Network built with TensorFlow/Keras.

🚀 Demo
Enter patient details → Get instant insurance cost prediction

📖 Overview
This project builds an end-to-end machine learning pipeline that:

Cleans and preprocesses the insurance dataset
Trains a deep neural network to predict insurance charges
Saves the trained model and scaler for deployment
Serves predictions via an interactive web app (Gradio / Streamlit)

📁 Project Structure
insurance-cost-prediction-app/
│
├── insurance.csv            # Dataset
├── model.py                 # Model training script
├── deployment.py            # Web app (Gradio/Streamlit)
├── insurance_model.h5       # Saved Keras model
├── scaler.pkl               # Saved StandardScaler
├── requirements.txt         # Dependencies
└── README.md

📊 Dataset
The dataset used is the Medical Cost Personal Dataset from Kaggle.

🧠 Model Architecture
Input Layer  →  128 neurons (ReLU)
             →  64 neurons  (ReLU)
             →  32 neurons  (ReLU)
Output Layer →  1 neuron    (Linear)

Optimizer: Adam
Loss: Mean Squared Error
Epochs: 100
Batch Size: 32

