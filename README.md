## Placement_prediction
# Student Placement Prediction Web App

This project is an end-to-end data science application that predicts a student's likelihood of securing a job placement based on their academic, personal, and experiential profile. It includes a deep data analysis, the training of a highly accurate machine learning model, and a user-friendly web interface built with Flask for live predictions and actionable feedback.

 
<!-- **Note to User:** Replace the link above with a GIF of your own application in action for a more professional look. -->

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Application](#how-to-run-the-application)
- [How to Use the App](#how-to-use-the-app)
- [Future Improvements](#future-improvements)

## Features

*   **Predictive Model**: Utilizes a sophisticated Stacking Ensemble model to achieve high accuracy (>95%) in predicting placement outcomes.
*   **Interactive Web Interface**: A clean and simple UI built with Flask and HTML/CSS allows users to input student data easily.
*   **Live Predictions**: Get instant placement probability and confidence scores.
*   **Actionable Feedback**: If a student is predicted as "Unlikely to be placed," the application provides personalized suggestions on key areas for improvement (e.g., CGPA, Internships, Projects).
*   **Data-Driven Insights**: The model is based on a thorough analysis of over 3000 student records, identifying the most influential factors for placement success.


## Methodology

The project follows a standard machine learning lifecycle:

1.  **Data Analysis & EDA**: The initial dataset was cleaned, and a deep exploratory data analysis was performed to understand feature distributions and correlations with the target variable (`Placement`).
2.  **Feature Engineering**: New, more predictive features like `Academic_Score` and `Practical_Score` were created to improve model performance.
3.  **Model Training & Tuning**: Several models were evaluated. A **Stacking Classifier** was chosen for its robustness, combining the strengths of a RandomForest and XGBoost model with a Logistic Regression meta-model.
4.  **Deployment**: The final model and data scaler were saved using `joblib` and integrated into a Flask web application, creating a user-facing tool for predictions.

## Technologies Used

*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy
*   **Frontend**: HTML, CSS
*   **Data Analysis & Visualization**: Matplotlib, Seaborn


## ML application to predict the chances of placement by using certain parameters

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```
https://github.com/hussainali99a/placement_prediction.git
```

### Create Virtual environment
```
python -m venv venv
.\venv\Scripts\activate
```

### Install required libraries
```
pip install Flask pandas numpy scikit-learn xgboost joblib
```

### Run the Application
```
python app.py
```
