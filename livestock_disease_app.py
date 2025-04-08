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

# Define early stage symptoms per disease (you can update with your knowledge)
early_warning_symptoms = {
    'foot and mouth': ['fever', 'blisters on mouth', 'lameness'],
    'anthrax': ['high fever', 'swelling under jaw', 'sudden death'],
    'lumpy virus': ['lumps on skin', 'fever'],
    'pneumonia': ['labored breathing', 'nasal discharge'],
    'blackleg': ['swollen muscle', 'limping'],
}

# List of symptoms based on training
symptom_options = list(mlb.classes_)

# Sidebar input
st.sidebar.header("📝 Input Animal Info")

age = st.sidebar.slider("Age (years)", min_value=0, max_value=20, value=2)
temperature = st.sidebar.slider("Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1, value=38.5)

# Symptom selector
symptoms = st.sidebar.multiselect("Select Observed Symptoms", options=symptom_options)

# Submit button
if st.sidebar.button("Predict Disease"):

    # Encode symptom input
    symptom_vector = np.zeros(len(mlb.classes_))
    for symptom in symptoms:
        if symptom in mlb.classes_:
            idx = list(mlb.classes_).index(symptom)
            symptom_vector[idx] = 1

    input_df = pd.DataFrame([[age, temperature] + list(symptom_vector)],
                            columns=['Age', 'Temperature'] + list(mlb.classes_))

    # Run prediction
    try:
        predicted_class = model.predict(input_df)[0]
        predicted_probs = model.predict_proba(input_df)[0]
        class_index = list(model.classes_).index(predicted_class)
        predicted_confidence = predicted_probs[class_index]

        # Show main results
        st.success(f"🦠 Predicted Disease: **{predicted_class}**")
        st.info(f"Confidence Level: **{predicted_confidence*100:.2f}%**")

        # Stage-based alert
        if predicted_confidence < 0.4:
            st.warning("🔍 Low confidence prediction. Symptoms may suggest an early or unclear stage. Monitor the animal closely.")
        elif 0.4 <= predicted_confidence < 0.75:
            st.info("✅ Symptoms suggest **early stage of disease**. Consider early treatment and isolation.")
        else:
            st.error("⚠️ High confidence — symptoms likely represent a **progressed stage** of disease. Immediate action is recommended.")

        # Cross-check early symptoms
        if predicted_class in early_warning_symptoms:
            expected_symptoms = set(early_warning_symptoms[predicted_class])
            matched = expected_symptoms.intersection(set(symptoms))
            if 0 < len(matched) < len(expected_symptoms):
                st.info(f"🧪 Early signs of **{predicted_class}** detected based on symptoms: {', '.join(matched)}")
            elif len(matched) == len(expected_symptoms):
                st.success(f"✅ All early indicators of **{predicted_class}** present.")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))

# Footer
st.markdown("---")
st.caption("Created by francis ❤️ for farmers to detect livestock diseases early and take action.")
