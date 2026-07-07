import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar el dataset
df = pd.read_csv("dataset/Model_creditoPersonal.csv")

# Eliminar columnas que no se usan
df = df.drop(columns=["ID", "ZIP Code"])

# Variables de entrada (X)
X = df.drop(columns=["Personal Loan"])

# Variable objetivo (y)
y = df["Personal Loan"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Crear modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrenar
modelo.fit(X_train, y_train)

# Evaluar
precision = modelo.score(X_test, y_test)

print(f"Precisión del modelo: {precision*100:.2f}%")

# Guardar el modelo
joblib.dump(modelo, "modelo.pkl")

print("Modelo entrenado y guardado correctamente.")