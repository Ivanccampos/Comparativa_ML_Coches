import streamlit as st

import joblib
import pandas as pd

# Cargar preprocesador y modelo
preprocessor = joblib.load("preprocessor2.joblib")
model = joblib.load("best_model2.joblib")

def pedir_datos():
    print("=== PREDICCIÓN DE PRECIO DE COCHE ===")

    model_car = input("Modelo del coche (ej: A Class): ")
    year = int(input("Año: "))
    transmission = input("Transmisión (Automatic/Manual/Semi-Auto): ")
    mileage = int(input("Kilometraje: "))
    fuelType = input("Combustible (Petrol/Diesel/Hybrid/Electric): ")
    tax = int(input("Impuesto (€): "))
    mpg = float(input("Consumo MPG: "))
    engineSize = float(input("Tamaño del motor (ej: 2.0): "))

    # DataFrame con las columnas EXACTAS del entrenamiento
    datos = pd.DataFrame([{
        "model": model_car,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuelType,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize
    }])

    return datos

def main():
    datos = pedir_datos()

    # Preprocesar
    datos_procesados = preprocessor.transform(datos)

    # Predecir precio
    precio = model.predict(datos_procesados)[0]

    print("\n💰 Precio estimado del coche:")
    print(f"{precio:,.2f} €")

if __name__ == "__main__":
    main()

