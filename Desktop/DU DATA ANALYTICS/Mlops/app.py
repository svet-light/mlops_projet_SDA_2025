import pickle
from flask import Flask, render_template, request
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Charger le modèle 

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        credit_lines_outstanding = float(request.form['credit_lines_outstanding'])
        loan_amt_outstanding = float(request.form['loan_amt_outstanding'])
        total_debt_outstanding = float(request.form['total_debt_outstanding'])
        income = float(request.form['income'])
        years_employed = float(request.form['years_employed'])
        fico_score = float(request.form['fico_score'])

        # Créer un DataFrame avec les données saisies
        data = pd.DataFrame({
            'credit_lines_outstanding': [credit_lines_outstanding],
            'loan_amt_outstanding': [loan_amt_outstanding],
            'total_debt_outstanding': [total_debt_outstanding],
            'income': [income],
            'years_employed': [years_employed],
            'fico_score': [fico_score]
        })

        # Faire la prédiction
        prediction = model.predict(data)
        result = "Défaut" if prediction[0] == 1 else "Non-Défaut"

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)