import joblib
import pandas as pd
import streamlit as st

# ------------------------------
# Cargar preprocesador y modelo
# ------------------------------
preprocessor = joblib.load("preprocessor2.joblib")
model = joblib.load("best_model2.joblib")

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(page_title="Predicción Precio de Coche", layout="centered")
st.title("💰 Predicción de Precio de Coche")

# ------------------------------
# Definir opciones conocidas
# ------------------------------
# Estas deben coincidir con las categorías vistas en entrenamiento
model_options = ["A Class", "B Class", "C Class", "E Class", "S Class"]  # ejemplo
transmission_options = ["Automatic", "Manual", "Semi-Auto"]
fuel_options = ["Petrol", "Diesel", "Hybrid", "Electric"]

# ------------------------------
# Formulario de entrada
# ------------------------------
with st.form(key="car_form"):
    st.header("Introduce los datos del coche")

    model_car = st.selectbox("Modelo del coche", model_options)
    year = st.number_input("Año", min_value=1900, max_value=2055, value=2020, step=1)
    transmission = st.selectbox("Transmisión", transmission_options)
    mileage = st.number_input("Kilometraje", min_value=0, value=10000, step=1000)
    fuelType = st.selectbox("Combustible", fuel_options)
    tax = st.number_input("Impuesto (€)", min_value=0, value=200, step=10)
    mpg = st.number_input("Consumo MPG", min_value=0.0, value=30.0, step=0.1)
    engineSize = st.number_input("Tamaño del motor (ej: 2.0)", min_value=0.0, value=2.0, step=0.1)

    submit_button = st.form_submit_button(label="Predecir Precio")

# ------------------------------
# Predicción
# ------------------------------
if submit_button:
    # Crear DataFrame con EXACTAMENTE 8 columnas
    datos = pd.DataFrame({
        "model": [model_car],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    try:
        # Preprocesar
        datos_procesados = preprocessor.transform(datos)

        # Predecir precio
        precio = model.predict(datos_procesados)[0]

        # Mostrar resultado
        st.success(f"💰 Precio estimado del coche: {precio:,.0f} €")

    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
        st.info("Asegúrate de usar solo opciones válidas de los desplegables.")

