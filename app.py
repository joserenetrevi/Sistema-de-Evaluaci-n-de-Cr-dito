import webbrowser
from threading import Timer
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("modelo.pkl")


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():

    # Obtener datos del formulario
    age = int(request.form["age"])
    experience = int(request.form["experience"])
    income = float(request.form["income"])
    family = int(request.form["family"])
    ccavg = float(request.form["ccavg"])
    education = int(request.form["education"])
    mortgage = float(request.form["mortgage"])
    securities = int(request.form["securities"])
    cd = int(request.form["cd"])
    online = int(request.form["online"])
    creditcard = int(request.form["creditcard"])

    # Crear DataFrame
    datos = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities,
        "CD Account": cd,
        "Online": online,
        "CreditCard": creditcard
    }])

    # Predicción
    prediccion = modelo.predict(datos)[0]

    # Probabilidad
    probabilidad = modelo.predict_proba(datos)[0]

    porcentaje = round(max(probabilidad) * 100, 2)

    if prediccion == 1:
        resultado = "APROBADO"
        color = "green"
    else:
        resultado = "RECHAZADO"
        color = "red"

    return render_template(
        "resultado.html",
        resultado=resultado,
        porcentaje=porcentaje,
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)