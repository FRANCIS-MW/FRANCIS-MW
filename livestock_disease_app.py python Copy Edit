import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import zipfile

# --- Load and train model ---
@st.cache_data
def load_and_train_model():
    with zipfile.ZipFile('animal_disease_dataset.csv.zip', 'r') as zip_ref:
        zip_ref.extractall()
    data = pd.read_csv('animal_disease_dataset.csv')

    # Combine symptoms
    data['Symptoms'] = data[['Symptom 1', 'Symptom 2', 'Symptom 3']].values.tolist()

    # Early stage indicator
    early_symptom_list = ['fever', 'loss of appetite', 'weakness']
    data['Early_Symptom'] = data['Symptoms'].apply(
        lambda symptoms: any(symptom in early_symptom_list for symptom in symptoms)
    )

    # Encode symptoms
    mlb = MultiLabelBinarizer()
    symptom_encoded = mlb.fit_transform(data['Symptoms'])
    symptom_data = pd.DataFrame(symptom_encoded, columns=mlb.classes_)

    # Combine features
    features = pd.concat([data[['Age', 'Temperature']], symptom_data], axis=1)
    labels = data['Disease']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, mlb, early_symptom_list

# --- Load model and symptom list ---
model, mlb, early_symptom_list = load_and_train_model()

# --- Streamlit UI ---
st.title("🐄 Livestock Disease Predictor")
st.markdown("Predict livestock diseases early using symptoms, temperature, and age.")

age = st.number_input("Enter animal's age (in years)", min_value=0.0, step=0.1, value=2.0)
temperature = st.number_input("Enter animal's temperature (°C)", min_value=30.0, max_value=45.0, value=38.0)
symptoms_input = st.text_input("Enter symptoms (comma-separated)", value="fever, weakness")

if st.button("Predict Disease"):
    symptoms = [s.strip().lower() for s in symptoms_input.split(',')]

    # Early stage detection
    is_early = any(symptom in early_symptom_list for symptom in symptoms)
    if is_early:
        st.success("🟢 Symptoms suggest early stage of disease.")
    else:
        st.warning("🟡 Symptoms may be late stage or unclear.")

    # Encode symptoms
    symptom_vector = np.zeros(len(mlb.classes_))
    for s in symptoms:
        if s in mlb.classes_:
            symptom_vector[list(mlb.classes_).index(s)] = 1

    # Prepare input
    input_data = pd.DataFrame([[age, temperature] + list(symptom_vector)],
                              columns=['Age', 'Temperature'] + list(mlb.classes_))

    # Predict
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
