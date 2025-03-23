from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle avec pickle
with open('loan_eligibility_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    data = request.form.to_dict()
    
    # Convertir les données en tableau numpy
    features = np.array([float(data['credit_lines_outstanding']),
                         float(data['loan_amt_outstanding']),
                         float(data['total_debt_outstanding']),
                         float(data['income']),
                         float(data['years_employed']),
                         float(data['fico_score'])]).reshape(1, -1)
    
    # Faire la prédiction
    prediction = model.predict(features)
    
    # Convertir la prédiction en résultat lisible
    result = 'Éligible' if prediction[0] == 1 else 'Non éligible'
    
    return render_template('index.html', prediction_text=f'Résultat de la prédiction: {result}')

if __name__ == '__main__':
    app.run(debug=True)