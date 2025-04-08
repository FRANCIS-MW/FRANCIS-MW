from flask import Flask, request, render_template_string
import pandas as pd
import joblib

# Load model and symptom encoder
model = joblib.load("model(1).pkl")
mlb = joblib.load("mlb.pkl")

app = Flask(__name__)

html_template = """
<!doctype html>
<title>Animal Disease Predictor</title>
<h2>🐄 Animal Disease Early Detection</h2>
<form method=post>
  Age: <input type=number name=age required><br><br>
  Temperature: <input type=number step=0.1 name=temp required><br><br>
  Symptoms (comma-separated): <input type=text name=symptoms required><br><br>
  <input type=submit value=Predict>
</form>

{% if prediction %}
  <h3>🩺 Predicted Disease: <span style="color:green">{{ prediction }}</span></h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        age = float(request.form["age"])
        temp = float(request.form["temp"])
        symptoms = [s.strip().lower() for s in request.form["symptoms"].split(",")]

        input_dict = {sym: 0 for sym in mlb.classes_}
        for s in symptoms:
            if s in input_dict:
                input_dict[s] = 1

        input_data = [age, temp] + list(input_dict.values())
        prediction = model.predict([input_data])[0]

    return render_template_string(html_template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
