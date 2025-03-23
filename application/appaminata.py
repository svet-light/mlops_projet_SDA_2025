import streamlit as st
import pandas as pd
import mlflow
import joblib



def load_Loan_Data():
    df = pd.read_csv("../mlflow_experiments/Loan_Data.csv", nrows=1) 
    return df


data = load_Loan_Data()
features = list(data.columns)

mlflow.set_tracking_uri("http://localhost:5001")

model_uri = "runs:/42cbaee966be49c2a0092ce3eb6a15aa/model"
model = mlflow.sklearn.load_model(model_uri)



# Les colonnes utilisées dans le modèle (sans customer_id et default)
features = [ 'credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score' ]  # adapte avec tes vraies colonnes

# Ajout de customer_id séparément
st.title("Prédiction de risque avec Logistic Regression")
st.write("Remplissez les valeurs des caractéristiques pour obtenir une prédiction:")

# Récupérer customer_id
customer_id = st.text_input("Customer ID")

# Récupérer les features du modèle
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

# Prédire quand on clique
if st.button("Prédire"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    # Afficher avec customer_id
    st.write(f"### Résultat pour client {customer_id} : {'Danger' if prediction == 1 else 'Bon'} (Probabilité : {proba:.2f})")

