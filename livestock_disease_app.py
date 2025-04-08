import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and encoder
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    mlb = joblib.load('model.pkl')
    return model, mlb

model, mlb = load_model()

# Early stage symptoms list
early_symptom_list = ['fever', 'loss of appetite', 'weakness']

# --- Streamlit UI ---
st.title("🐄 Livestock Disease Predictor (Using Saved Model)")
st.markdown("Predict livestock diseases early using symptoms, temperature, and age.")

age = st.number_input("Enter animal's age (in years)", min_value=0.0, step=0.1, value=2.0)
temperature = st.number_input("Enter animal's temperature (°C)", min_value=30.0, max_value=45.0, value=38.0)
symptoms_input = st.text_input("Enter symptoms (comma-separated)", value="fever, weakness")

if st.button("Predict Disease"):
    symptoms = [s.strip().lower() for s in symptoms_input.split(',')]

    # Early stage check
    is_early = any(symptom in early_symptom_list for symptom in symptoms)
    if is_early:
        st.success("🟢 Symptoms suggest early stage of disease.")
    else:
        st.warning("🟡 Symptoms may be late stage or unclear.")

    # Encode input symptoms
    symptom_vector = np.zeros(len(mlb.classes_))
    for s in symptoms:
        if s in mlb.classes_:
            symptom_vector[list(mlb.classes_).index(s)] = 1

    # Prepare input
    input_data = pd.DataFrame([[age, temperature] + list(symptom_vector)],
                              columns=['Age', 'Temperature'] + list(mlb.classes_))

    # Prediction
    predicted_class = model.predict(input_data)[0]
    predicted_prob = model.predict_proba(input_data)[0]
    class_index = list(model.classes_).index(predicted_class)
    predicted_class_prob = predicted_prob[class_index]

    # Output
    early_threshold = 0.7
    if predicted_class_prob > early_threshold:
        st.success(f"✅ Early detection: **{predicted_class}** likely.\nProbability: {predicted_class_prob:.2f}")
    else:
        st.warning(f"⚠️ Disease **{predicted_class}** detected.\nProbability: {predicted_class_prob:.2f}")
    
    st.markdown("📞 Please consult a vet for treatment.")
