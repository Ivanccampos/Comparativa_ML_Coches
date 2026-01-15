import streamlit as st
import pandas as pd
import joblib
import warnings

# ------------------------------
# Cargar modelo y preprocesador
# ------------------------------
MODEL_PATH = "/content/drive/MyDrive/models/best_model2.joblib"
PREPROCESSOR_PATH = "/content/drive/MyDrive/models/preprocessor2.joblib"

loaded_model = joblib.load(MODEL_PATH)
loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)

st.title("💰 Predicción de Precio de Coche (Random Forest)")

# ------------------------------
# Opciones conocidas (del entrenamiento)
# ------------------------------
model_options = ["A Class", "B Class", "C Class", "E Class", "S Class", "GLE Class"]
transmission_options = ["Automatic", "Manual", "Semi-Auto"]
fuel_options = ["Petrol", "Diesel", "Hybrid", "Electric"]

# ------------------------------
# Formulario de entrada
# ------------------------------
with st.form(key="car_form"):
    st.header("Introduce las características del vehículo")
    
    user_model = st.selectbox("Modelo del coche", model_options)
    user_year = st.number_input("Año", min_value=1900, max_value=2055, value=2020, step=1)
    user_transmission = st.selectbox("Transmisión", transmission_options)
    user_mileage = st.number_input("Kilometraje", min_value=0, value=10000, step=1000)
    user_fuel_type = st.selectbox("Tipo de combustible", fuel_options)
    user_tax = st.number_input("Impuesto (€)", min_value=0.0, value=200.0, step=10.0)
    user_mpg = st.number_input("Consumo MPG", min_value=0.0, value=30.0, step=0.1)
    user_engineSize = st.number_input("Tamaño del motor (ej: 2.0)", min_value=0.0, value=2.0, step=0.1)
    
    submit_button = st.form_submit_button(label="Predecir Precio")

# ------------------------------
# Predicción
# ------------------------------
if submit_button:
    # Crear DataFrame exactamente igual al usado en entrenamiento
    vehicle_features = pd.DataFrame([{
        "model": user_model,
        "year": user_year,
        "transmission": user_transmission,
        "mileage": user_mileage,
        "fuelType": user_fuel_type,
        "tax": user_tax,
        "mpg": user_mpg,
        "engineSize": user_engineSize
    }])

    try:
        # Predecir precio ignorando warnings por categorías desconocidas
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module='sklearn.preprocessing._encoders'
            )
            predicted_price = loaded_model.predict(vehicle_features)[0]

        st.success(f"💰 Precio estimado del vehículo: {predicted_price:,.0f} €")

    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
        st.info("Asegúrate de seleccionar valores válidos de los desplegables.")

