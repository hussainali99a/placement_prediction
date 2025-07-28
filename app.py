# app.py (Final Enhanced Version with Stacking Model)
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load all necessary components ---
model = joblib.load('placement_model_stacked.joblib')
scaler = joblib.load('scaler_engineered.joblib')
feature_list = joblib.load('feature_list.joblib')

# Define ideal thresholds for giving feedback
IDEAL_THRESHOLDS = {
    'CGPA': 8.5,
    'Projects_Completed': 3,
    'Communication_Skills': 8,
    'Internship_Experience': 1
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # --- 1. Get raw input values ---
    raw_inputs = {
        'IQ': float(request.form['iq']),
        'CGPA': float(request.form['cgpa']),
        'Prev_Sem_Result': float(request.form['cgpa']), # Use CGPA as a proxy if not provided
        'Internship_Experience': int(request.form['internship']),
        'Projects_Completed': int(request.form['projects']),
        'Communication_Skills': int(request.form['communication'])
    }
    
    # --- 2. Engineer features from raw inputs ---
    engineered_features = {
        'IQ': raw_inputs['IQ'],
        'CGPA': raw_inputs['CGPA'],
        'Internship_Experience': raw_inputs['Internship_Experience'],
        'Projects_Completed': raw_inputs['Projects_Completed'],
        'Communication_Skills': raw_inputs['Communication_Skills'],
        'Academic_Score': (raw_inputs['CGPA'] * 0.6) + (raw_inputs['Prev_Sem_Result'] * 0.4),
        'Practical_Score': (raw_inputs['Internship_Experience'] * 3) + raw_inputs['Projects_Completed'],
        'IQ_x_CGPA': raw_inputs['IQ'] * raw_inputs['CGPA']
    }
    
    # --- 3. Create the final feature array in the correct order ---
    final_features_list = [engineered_features[feature] for feature in feature_list]
    features_array = np.array(final_features_list).reshape(1, -1)
    
    # --- 4. Scale and Predict ---
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    
    # --- 5. Generate Dynamic Feedback ---
    output_text = ""
    confidence = ""
    suggestions = []
    
    if prediction[0] == 1:
        output_text = "High Placement Probability"
        confidence = f"Confidence: {probability[0][1]*100:.2f}%"
        suggestions = [
            "This profile is strong and meets key industry benchmarks.",
            "Focus on targeted company preparation and acing the interviews.",
            "Highlight the completed projects and internship experience on your resume."
        ]
    else:
        output_text = "Low Placement Probability"
        confidence = f"Confidence: {probability[0][0]*100:.2f}%"
        
        # Give more specific, actionable feedback
        if raw_inputs['CGPA'] < IDEAL_THRESHOLDS['CGPA']:
            suggestions.append(f"Improve CGPA: At {raw_inputs['CGPA']:.2f}, this is a critical area. Aim for {IDEAL_THRESHOLDS['CGPA']}+ to pass initial screenings.")
        
        if raw_inputs['Internship_Experience'] == 0:
            suggestions.append("Gain Internship Experience: This is the most impactful practical step. Lack of an internship significantly lowers placement chances.")
        
        if raw_inputs['Projects_Completed'] < IDEAL_THRESHOLDS['Projects_Completed']:
            suggestions.append(f"Build More Projects: Having {raw_inputs['Projects_Completed']} project(s) is a start. Build more complex projects to showcase technical skills.")

        if raw_inputs['Communication_Skills'] < IDEAL_THRESHOLDS['Communication_Skills']:
            suggestions.append(f"Boost Communication Skills: A score of {raw_inputs['Communication_Skills']} may not be sufficient for competitive interviews. Practice public speaking and mock interviews.")
            
        if not suggestions:
            suggestions.append("While no single area is critically low, the combination of factors makes this profile less competitive. A general improvement across all areas is advised.")

    return render_template('index.html', 
                           prediction_text=output_text, 
                           confidence_text=confidence,
                           suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)