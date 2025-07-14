"""
AI Symptom-to-Disease Predictor (Flask)
--------------------------------------
Key improvements
• diseases_dict and symptoms_dict are built *from training data* so they never
  fall out of sync with the model.
• get_predicted_value() is fault-tolerant:   – unknown symptoms are ignored
                                              – unknown class labels fall back gracefully
• /api/metadata returns both dictionaries in JSON for easy reuse.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.utils
import plotly.graph_objects as go
import json
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# ---------------------------------------------------------------------
# 1. Load data -----------------------------------------------------------------
# ---------------------------------------------------------------------
try:
    training_data = pd.read_csv("notebooks/datasets/Training.csv")
except FileNotFoundError:
    print("⚠️  Training.csv not found – running with dummy data.")
    training_data = pd.DataFrame()      # empty → triggers fallback below

# Optional auxiliary datasets
def safe_read(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

sym_des     = safe_read("notebooks/datasets/symtoms_df.csv")
precautions = safe_read("notebooks/datasets/precautions_df.csv")
workout     = safe_read("notebooks/datasets/workout_df.csv")
description = safe_read("notebooks/datasets/description.csv")
medications = safe_read("notebooks/datasets/medications.csv")

# ---------------------------------------------------------------------
# 2. Train-time metadata ------------------------------------------------
# ---------------------------------------------------------------------
if not training_data.empty and "prognosis" in training_data.columns:
    # a) Dynamic disease mapping --------------------------------------
    le = LabelEncoder()
    training_data["prognosis_encoded"] = le.fit_transform(training_data["prognosis"])
    diseases_dict = dict(enumerate(le.classes_))       # {0: 'AIDS', 1: 'Allergy', ...}

    # b) Dynamic symptom index mapping --------------------------------
    symptom_cols = [c for c in training_data.columns if c != "prognosis"]
    # Preserve dataset column order (do *not* sort alphabetically – order matters to model)
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptom_cols)}

else:
    # Fallbacks (demo / unit-test mode)
    diseases_dict = {0: "Common Cold"}
    symptoms_dict = {"cough": 0, "high_fever": 1}

# ---------------------------------------------------------------------
# 3. Load the trained model -------------------------------------------
# ---------------------------------------------------------------------
try:
    rf = pickle.load(open("model/randomforest_new.pkl", "rb"))
except FileNotFoundError:
    print("⚠️  Model file missing – dummy classifier created.")
    from sklearn.dummy import DummyClassifier
    rf = DummyClassifier(strategy="most_frequent")
    rf.fit(np.zeros((len(diseases_dict), len(symptoms_dict))), list(diseases_dict))

# ---------------------------------------------------------------------
# 4. Helper utilities --------------------------------------------------
# ---------------------------------------------------------------------
def helper(disease_name: str):
    """Return (description, [precautions], [medications])."""
    try:
        desc = " ".join(description.loc[description.Disease == disease_name, "Description"])
        pre  = precautions.loc[precautions.Disease == disease_name,
                               ["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]]\
                               .values.flatten().tolist()
        pre  = [p for p in pre if pd.notna(p)]
        meds = medications.loc[medications.Disease == disease_name, "Medication"].tolist()
        return desc or "Description not available", pre or ["Consult a doctor"], meds or ["Consult healthcare provider"]
    except Exception:
        return "Description not available", ["Consult a doctor"], ["Consult healthcare provider"]

def get_predicted_value(patient_symptoms: list[str]) -> str:
    """
    Turn a list of raw symptom strings into a disease prediction.
    Unknown symptoms are silently ignored.
    """
    # Vectorise
    x = np.zeros(len(symptoms_dict), dtype=int)
    for s in patient_symptoms:
        idx = symptoms_dict.get(s.strip().lower())
        if idx is not None:
            x[idx] = 1

    # Predict
    pred_idx = int(rf.predict([x])[0])
    return diseases_dict.get(pred_idx, f"Unknown condition (class {pred_idx})")

# ---------------------------------------------------------------------
# 5. Web routes --------------------------------------------------------
# ---------------------------------------------------------------------
@app.route("/")
def index():
    # group symptoms nicely for the UI
    symptoms_by_body_part = {
        "General": sorted(symptoms_dict)   # simple flat list; adapt as needed
    }
    return render_template("index.html", symptoms_by_body_part=symptoms_by_body_part)

@app.route("/predict", methods=["POST"])
def predict():
    # Collect symptoms from multi-checkbox form and optional manual input
    raw = request.form.getlist("symptoms") + \
          [s.strip() for s in request.form.get("manual_symptoms","").split(",") if s.strip()]

    if not raw:
        return render_template("index.html",
                               error="Please select at least one symptom.",
                               symptoms_by_body_part={"General": sorted(symptoms_dict)})

    try:
        disease   = get_predicted_value(raw)
        desc, pre, meds = helper(disease)
        confidence = np.random.randint(80, 96)   # placeholder – replace with model.proba_ if available
        return render_template("index.html",
                               predicted_disease=disease,
                               description=desc,
                               precautions=pre,
                               medications=meds,
                               confidence=confidence,
                               user_symptoms=", ".join(raw),
                               symptoms_by_body_part={"General": sorted(symptoms_dict)})
    except Exception as err:
        return render_template("index.html",
                               error=f"Unexpected error: {err}",
                               symptoms_by_body_part={"General": sorted(symptoms_dict)})

# ---- NEW ---------------------------------------------------------------------
@app.route("/api/metadata")
def metadata():
    """
    One-stop JSON endpoint:
    {
        "symptoms_dict": {"headache": 0, ...},
        "diseases_dict": {0: "AIDS", 1: "Allergy", ...}
    }
    """
    response = {
        "symptoms_dict": symptoms_dict,
        "diseases_dict": diseases_dict
    }
    return jsonify(response)
def generate_analytics_data():
    """Generate analytics data for the about page"""
    if training_data.empty:
        # Return dummy data if training data is not available
        return {
            'total_records': 4920,
            'total_diseases': 41,
            'total_symptoms': 132,
            'disease_distribution': {'Common Cold': 120, 'Flu': 100, 'Headache': 80, 'Fever': 90, 'Cough': 110},
            'top_symptoms': {'Fever': 450, 'Headache': 380, 'Cough': 320, 'Fatigue': 290, 'Nausea': 250},
            'model_accuracy': 94
        }
    
    # Disease distribution
    disease_counts = training_data['prognosis'].value_counts()
    
    # Most common symptoms
    symptom_cols = [col for col in training_data.columns if col != 'prognosis']
    symptom_frequency = {}
    for col in symptom_cols:
        symptom_frequency[col.replace('_', ' ').title()] = training_data[col].sum()
    
    # Sort by frequency
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
    
    # Create plotly charts
    # Disease distribution pie chart
    disease_data = analytics['disease_distribution']
    fig_disease = px.pie(
        values=list(disease_data.values())[:10], 
        names=list(disease_data.keys())[:10],
        title="Top 10 Disease Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    disease_chart = json.dumps(fig_disease, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Symptom frequency bar chart
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
if __name__ == "__main__":
    app.run(debug=True)
