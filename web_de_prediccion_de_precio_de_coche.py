import joblib
import pandas as pd
import streamlit as st

# ------------------------------
# Cargar modelo (pipeline completo)
# ------------------------------
model = joblib.load("best_model5.joblib")  # contiene preprocesador + Random Forest

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(page_title="Predicción Precio de Coche", layout="centered")
st.title("💰 Predicción de Precio de Coche")

# ------------------------------
# Definir opciones conocidas
# ------------------------------
# Añade el espacio inicial a cada nombre de modelo
model_options = [
    " A Class", "B Class", "C Class", " E Class", 
    " CL Class", " GLC Class", " GLA Class", " GLE Class"
]
transmission_options = ["Automatic", "Manual", "Semi-Auto", "Other"]
fuel_options = ["Petrol", "Diesel", "Hybrid", "Other"]

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
    engineSize = st.number_input("Tamaño del motor (ej: 2.0)", min_value=0.0, value=2.0, step=0.1)

    submit_button = st.form_submit_button(label="Predecir Precio")

# ------------------------------
# Predicción
# ------------------------------
if submit_button:
    try:
        # 1. Definir los datos en el ORDEN EXACTO del dataset original
        # El orden del CSV original es: model, year, price (se quita), transmission, mileage, fuelType, tax, mpg, engineSize
        
        datos_dict = {
            "model": [model_car],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "engineSize": [engineSize]
        }
        
        # 2. Crear el DataFrame
        datos_usuario = pd.DataFrame(datos_dict)

        # 3. Asegurar que las columnas están en el orden que espera el preprocesador
        # Basado en tu archivo comparativas_modelos_ml.py
        columnas_entrenamiento = ['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']
        datos_usuario = datos_usuario[columnas_entrenamiento]

        # 4. Predicción
        with st.spinner("Calculando..."):
            precio = model.predict(datos_usuario)[0]

        # Formatear precio a moneda europea
        precio_formateado = f"{precio:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        # Mostrar resultado
        st.success(f"💰 Precio estimado del coche: €{precio_formateado}")

    except Exception as e:
        st.error(f"Se produjo un error en la predicción: {e}")
        st.info("Asegúrate de usar solo opciones válidas de los desplegables.")

