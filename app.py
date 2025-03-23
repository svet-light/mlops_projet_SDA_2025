from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle
model = pickle.load(open("random_forest_model.pkl", "rb"))

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Récupération des valeurs du formulaire
        features = [
            float(request.form["credit_lines_outstanding"]),
            float(request.form["loan_amt_outstanding"]),
            float(request.form["total_debt_outstanding"]),
            float(request.form["income"]),
            float(request.form["years_employed"]),
            float(request.form["fico_score"])
        ]

        # Création d'un DataFrame pour la prédiction
        df = pd.DataFrame([features], columns=[
            "credit_lines_outstanding",
            "loan_amt_outstanding",
            "total_debt_outstanding",
            "income",
            "years_employed",
            "fico_score"
        ])

        # Prédiction binaire (0 ou 1)
        # prediction = model.predict(df)[0]
            
        # **Prédiction de la probabilité**
        probability = model.predict_proba(df)[0][1]  # Probabilité de défaut

        # Création du message basé sur la prédiction et la probabilité
        message = f"Risque de défaut: {probability*100:.2f}%."


        return render_template("index.html", prediction_text=message)


    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
