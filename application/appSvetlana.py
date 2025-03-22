from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load


app = Flask(__name__)
model = load("random_forest_model.pkl")


def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["Number of Open Credit Lines"])
        loan_amt_outstanding = float(request.form["Outstanding Loan Amount"])
        total_debt_outstanding = float(request.form["Total Outstanding Debt"])
        income = float(request.form["Annual Income"])
        years_employed = int(request.form["Years of Employment"])
        fico_score = int(request.form["FICO Credit Score"])
        #prediction = model.predict(
        #    [[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
        #)
        prediction = model.predict_proba(np.array([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]))


        #if prediction[0] == 1:
        return render_template(
            "index.html",
            prediction_text = f"Probability of Default: {prediction[0][1]:.4f}"
        )

        #else:
        #    return render_template(
        #        "index.html", prediction_text="You are well. No worries :)"
        #    )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
