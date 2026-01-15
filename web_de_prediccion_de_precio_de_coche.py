import joblib
import pandas as pd
import streamlit as st

# Cargar preprocesador y modelo
preprocessor = joblib.load("preprocessor2.joblib")
model = joblib.load("best_model2.joblib")

# Título de la app
st.title("💰 Predicción de Precio de Coche")

# Formulario de entrada de datos
with st.form(key="car_form"):
    st.header("Introduce los datos del coche")

    model_car = st.text_input("Modelo del coche (ej: A Class)")
    
    year = st.number_input("Año", min_value=1900, max_value=2050, value=2020, step=1)
    
    transmission = st.selectbox("Transmisión", ["Automatic", "Manual", "Semi-Auto"])
    
    mileage = st.number_input("Kilometraje", min_value=0, value=10000, step=1000)
    
    fuelType = st.selectbox("Combustible", ["Petrol", "Diesel", "Hybrid", "Electric"])
    
    tax = st.number_input("Impuesto (€)", min_value=0, value=200, step=10)
    
    mpg = st.number_input("Consumo MPG", min_value=0.0, value=30.0, step=0.1)
    
    engineSize = st.number_input("Tamaño del motor (ej: 2.0)", min_value=0.0, value=2.0, step=0.1)
    
    submit_button = st.form_submit_button(label="Predecir Precio")

# Cuando el usuario envía el formulario
if submit_button:
    # Crear DataFrame con las columnas EXACTAS del entrenamiento
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

    # Preprocesar
    datos_procesados = preprocessor.transform(datos)

    # Predecir precio
    precio = model.predict(datos_procesados)[0]

    # Mostrar resultado
    st.success(f"💰 Precio estimado del coche: {precio:,.0f} €")
