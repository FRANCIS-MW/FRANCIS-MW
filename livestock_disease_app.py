import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set app title
st.set_page_config(page_title="Livestock Disease Predictor", layout="centered")
st.title("🐄 Livestock Disease Prediction App with Early Detection")

# Load model and symptom encoder
model = joblib.load("model.pkl")
mlb = joblib.load("mlb.pkl")

# Define early warning symptoms (example)
early_warning_symptoms = {
    'foot and mouth': ['fever', 'blisters on mouth', 'lameness'],
    'anthrax': ['high fever', 'swelling under jaw', 'sudden death'],
    'lumpy virus': ['lumps on skin', 'fever'],
    'pneumonia': ['labored breathing', 'nasal discharge'],
    'blackleg': ['swollen muscle', 'limping'],
}

# List of symptoms based on what was used during training
symptom_options = list(mlb.classes_)

# Sidebar input
st.sidebar.header("📝 Input Animal Information")

age = st.sidebar.slider("Age (in years)", min_value=0, max_value=20, value=2)
temperature = st.sidebar.slider("Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1, value=38.5)

# Multi-select for symptoms
symptoms = st.sidebar.multiselect("Select Observed Symptoms", options=symptom_options)

# Submit button
if st.sidebar.button("Predict Disease"):

    # Encode symptoms into the same structure used in training
    symptom_vector = np.zeros(len(mlb.classes_))
    for symptom in symptoms:
        if symptom in mlb.classes_:
            index = list(mlb.classes_).index(symptom)
            symptom_vector[index] = 1

    # Construct input DataFrame with correct column order
    input_data = pd.DataFrame([[age, temperature] + list(symptom_vector)],
                              columns=['Age', 'Temperature'] + list(mlb.classes_))

    # Predict disease
    try:
        predicted_class = model.predict(input_data)[0]
        predicted_probs = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(predicted_class)
        predicted_confidence = predicted_probs[class_index]

        # Display results
        st.success(f"🦠 Predicted Disease: **{predicted_class}**")
        st.info(f"Confidence: **{predicted_confidence*100:.2f}%**")

        # --- EARLY WARNING LOGIC ---
        if predicted_class in early_warning_symptoms:
            early_signs = set(early_warning_symptoms[predicted_class])
            match_count = len(early_signs.intersection(set(symptoms)))
            if 0 < match_count < len(early_signs):
                st.warning(f"⚠️ **Early signs of {predicted_class}** detected. Consider monitoring and consulting a vet.")
            elif match_count == len(early_signs):
                st.info("✅ Full symptom pattern detected. This might be an advanced case.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Ensure input format matches training data features.")

# Footer
st.markdown("---")
st.caption("Created with ❤️ for farmers to detect livestock diseases early.")
