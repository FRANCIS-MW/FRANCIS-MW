import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set app title
st.set_page_config(page_title="Livestock Disease Predictor", layout="centered")
st.title("🐄 Livestock Disease Prediction App")

# Load model and symptom encoder
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")

# List of symptoms based on what was used during training
symptom_options = list(mlb.classes_)

# Sidebar input
st.sidebar.header("📝 Input Animal Information")

age = st.sidebar.slider("Age (in years)", min_value=0, max_value=20, value=2)
temperature = st.sidebar.slider("Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1, value=38.5)

# Multi-select for symptoms
symptoms = st.sidebar.multiselect("Select Symptoms", options=symptom_options)

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
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Ensure input format matches training data features.")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit")
