import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Precio de coche", layout="centered")

st.title("🚗 Predicción del precio de un coche")
st.header('amongus', divide='rainbow')

# Cargar modelo y preprocesador
model = joblib.load('best_model1.joblib')
preprocessor = joblib.load('preprocessor1.joblib')

st.header("Introduce los datos del coche")

year = st.number_input("Año de fabricación", min_value=1990, max_value=2025, step=1)
km = st.number_input("Kilómetros", min_value=0, step=1000)
fuel = st.selectbox("Combustible", ["Petrol", "Diesel", "Electric"])
transmission = st.selectbox("Transmisión", ["Manual", "Automatic"])
brand = st.text_input("Marca")

if st.button("Calcular precio"):
    data = {
        'year': [year],
        'km': [km],
        'fuel': [fuel],
        'transmission': [transmission],
        'brand': [brand]
    }

    df = pd.DataFrame(data)
    X = preprocessor.transform(df)
    precio = model.predict(X)[0]

    st.success(f"💰 Precio estimado: {precio:,.2f} €")
