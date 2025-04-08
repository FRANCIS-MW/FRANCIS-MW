import streamlit as st
import pandas as pd

# Load model and symptom encoder
model = joblib.load("model(1).pkl")
mlb = joblib.load("mlb.pkl")

# Streamlit UI
st.title("🐄 Animal Disease Early Detection App")
st.markdown("Provide basic animal info and symptoms for disease prediction.")

age = st.number_input("Age", min_value=0.0, step=0.1)
temp = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, step=0.1)
symptom_input = st.text_input("Symptoms (comma-separated)", placeholder="e.g. coughing, loss of appetite")

if st.button("Predict"):
    if symptom_input:
        symptoms = [s.strip().lower() for s in symptom_input.split(",")]
        input_dict = {sym: 0 for sym in mlb.classes_}
        for s in symptoms:
            if s in input_dict:
                input_dict[s] = 1

        input_data = [age, temp] + list(input_dict.values())
        prediction = model.predict([input_data])[0]

        st.success(f"🩺 Predicted Disease: **{prediction}**")
    else:
        st.warning("Please enter at least one symptom.")
