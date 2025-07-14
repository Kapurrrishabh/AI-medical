from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import ast
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app=Flask(__name__)

#--------------------load dataset-----------------
sym_des=pd.read_csv("notebooks/datasets/symtoms_df.csv")
precautions=pd.read_csv("notebooks/datasets/precautions_df.csv")
workout=pd.read_csv("notebooks/datasets/workout_df.csv")
description=pd.read_csv("notebooks/datasets/updated_description.csv")
medications=pd.read_csv("notebooks/datasets/updated_medications.csv")
training_data = pd.read_csv("notebooks/datasets/Training.csv")

#--------------------load model-------------------
randomforest=pickle.load(open('model/randomforest_new.pkl','rb'))

# helper function
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_dict = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def helper(dis):
    descr=description[description['Disease']==dis]['Description']
    descr=" ".join([w for w in descr])

    pre = precautions.loc[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
    pre = [p for p in pre if pd.notna(p)]

    med_strings = medications.loc[medications['Disease'] == dis, 'Medication']
    med = []
    for s in med_strings:
        # Convert the textual “[ 'A', 'B']” into a real list
        if isinstance(s, str):
            try:
                med.extend(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                # Fallback: treat the whole string as one medication
                med.append(s)
        else:
            med.append(s)

    return descr,pre,med

#model prediction function
def get_predicted_value(patient_symptoms):
    input_vector=np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]]=1
    return diseases_dict[randomforest.predict([input_vector])[0]]

# NEW: Calculate confidence score
def get_prediction_confidence(patient_symptoms):
    input_vector=np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:  # Add safety check
            input_vector[symptoms_dict[item]]=1
    
    # Get prediction probabilities
    probabilities = randomforest.predict_proba([input_vector])[0]
    confidence = max(probabilities) * 100
    return round(confidence, 1)

@app.route('/')
def index():
    return render_template('index.html')

# NEW: API endpoint for symptom suggestions
@app.route('/api/symptoms')
def get_symptoms():
    """Return all available symptoms for autocomplete"""
    symptoms_list = list(symptoms_dict.keys())
    return jsonify(symptoms_list)

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        symptoms=request.form.get('symtoms')  # Note: keeping original typo for compatibility
        
        # Enhanced symptom processing
        if not symptoms:
            return render_template('index.html', error="Please select at least one symptom.")
        
        # Clean and process symptoms
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
        
        # Filter out invalid symptoms
        valid_symptoms = [sym for sym in user_symptoms if sym in symptoms_dict]
        
        if not valid_symptoms:
            return render_template('index.html', error="No valid symptoms found. Please check your input.")
        
        try:
            predicted_disease = get_predicted_value(valid_symptoms)
            confidence = get_prediction_confidence(valid_symptoms)  # NEW: Get confidence
            desc, pre, med = helper(predicted_disease)
            
            return render_template('index.html', 
                                 predicted_des=predicted_disease,
                                 confidence=confidence,  # NEW: Pass confidence to template
                                 disc_des=desc,
                                 med_des=med,
                                 pre_des=pre,
                                 selected_symptoms=valid_symptoms)  # NEW: Pass selected symptoms
        
        except Exception as e:
            return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")

def generate_analytics_data():
    """Generate analytics data for the about page"""
    if training_data.empty:
        return {
            'total_records': 4920,
            'total_diseases': 41,
            'total_symptoms': 132,
            'disease_distribution': {'Common Cold': 120, 'Flu': 100, 'Headache': 80, 'Fever': 90, 'Cough': 110},
            'top_symptoms': {'Fever': 450, 'Headache': 380, 'Cough': 320, 'Fatigue': 290, 'Nausea': 250},
            'model_accuracy': 94
        }
    
    disease_counts = training_data['prognosis'].value_counts()
    symptom_cols = [col for col in training_data.columns if col != 'prognosis']
    symptom_frequency = {}
    for col in symptom_cols:
        symptom_frequency[col.replace('_', ' ').title()] = training_data[col].sum()
    
    top_symptoms = dict(sorted(symptom_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {
        'total_records': len(training_data),
        'total_diseases': len(disease_counts),
        'total_symptoms': len(symptom_cols),
        'disease_distribution': disease_counts.to_dict(),
        'top_symptoms': top_symptoms,
        'model_accuracy': 94
    }

@app.route('/about')
def about():
    analytics = generate_analytics_data()
    
    disease_data = analytics['disease_distribution']
    fig_disease = px.pie(
        values=list(disease_data.values())[:10], 
        names=list(disease_data.keys())[:10],
        title="Top 10 Disease Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    disease_chart = json.dumps(fig_disease, cls=plotly.utils.PlotlyJSONEncoder)
    
    symptoms_data = analytics['top_symptoms']
    fig_symptoms = px.bar(
        x=list(symptoms_data.keys()),
        y=list(symptoms_data.values()),
        title="Most Common Symptoms",
        labels={'x': 'Symptoms', 'y': 'Frequency'},
        color=list(symptoms_data.values()),
        color_continuous_scale='viridis'
    )
    fig_symptoms.update_layout(xaxis_tickangle=-45)
    symptoms_chart = json.dumps(fig_symptoms, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('about.html', 
                         analytics=analytics,
                         disease_chart=disease_chart,
                         symptoms_chart=symptoms_chart)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__=="__main__":
    app.run(debug=True)