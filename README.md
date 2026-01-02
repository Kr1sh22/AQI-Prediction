# Hourly AQI Prediction Using Deep Learning (MLP & LSTM)

## 📌 Project Overview
This project focuses on **predicting the next-hour Air Quality Index (AQI)** for an Indian city using **deep learning time-series models**.  
The models are trained on **hourly pollutant data** derived from the *city_hour* dataset and use historical pollutant values to forecast future AQI.

The project implements and compares **two neural network architectures**:
- Multi-Layer Perceptron (MLP)
- Stacked Long Short-Term Memory (LSTM)

## 📂 Dataset
- **Dataset Name:** Air Quality Data in India  
- **Source:** Kaggle  
- **Dataset Link:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india  
- **File Used:** `city_hour.csv` (cleaned and filtered for city-level hourly data)

### Features Used (6 Total)
- PM2.5  
- PM10  
- O₃  
- CO  
- NO₂  
- AQI *(target variable)*

## 🎯 Problem Statement
Given the **last 6 hours of pollutant concentrations and AQI**, predict the **AQI for the next hour**.

This is framed as a **time-series regression problem**.

## 🧠 Models Implemented

### 1️⃣ Multi-Layer Perceptron (MLP)
- Input: Flattened time window (6 hours × 6 features = 36 inputs)
- Architecture:
  - Dense(128) → Dense(64) → Dense(32) → Dense(16)
  - ReLU activations
- Output: Next-hour AQI
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Regularization: Early Stopping

### 2️⃣ Stacked LSTM
- Input Shape: (6 timesteps, 6 features)
- Architecture:
  - LSTM(100, return_sequences=True)
  - LSTM(50)
  - Dropout layers
- Output: Next-hour AQI
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Regularization: Early Stopping

## ⚙️ Data Preprocessing
- City-level filtering (e.g., Delhi)
- Linear interpolation for missing values
- MinMax scaling applied to all features
- Sliding window sequence creation (6-hour lookback)

## 📊 Model Evaluation
Models are evaluated on unseen test data using:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

Evaluation is performed after inverse-scaling predicted AQI values.

## 🔮 Prediction Pipeline
Separate prediction scripts are provided for both models:
- Accept last 6 hours of pollutant + AQI data
- Scale inputs using saved scalers
- Predict next-hour AQI
- Map AQI value to standard AQI categories:
  - Good, Satisfactory, Moderate, Poor, Very Poor, Severe

## 💾 Saved Artifacts
- Trained model files (`.keras`)
- Fitted scalers (`.pkl`)
- Training history for loss visualization

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook / Google Colab

## 🚀 How to Run
1. Prepare cleaned city-level hourly data (`delhi_cleaned.csv`)
2. Run the training script (MLP or LSTM)
3. Saved models and scalers will be generated automatically
4. Run the corresponding prediction script to forecast next-hour AQI

## 📌 Use Case
- Time-series forecasting
- Environmental monitoring
- Deep learning model comparison
- Academic and portfolio demonstration project
