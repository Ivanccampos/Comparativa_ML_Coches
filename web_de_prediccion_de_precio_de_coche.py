import joblib
import pandas as pd
import streamlit as st

# ------------------------------
# Cargar modelo (pipeline completo)
# ------------------------------
model = joblib.load("best_model.joblib")

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(
    page_title="Predicción Precio de Coche",
    layout="centered"
)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://estaticos-cdn.prensaiberica.es/clip/ada6fbfb-ca1f-4641-ae31-a2488cc9208e_16-9-discover-aspect-ratio_default_0.webp");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
        [data-testid="stForm"] {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
     )

add_bg_from_url()

st.title("💰 Predicción de Precio de Coche")

# ------------------------------
# Opciones válidas (coherentes con el dataset)
# ------------------------------
model_options = [
    "A Class", "B Class", "C Class", "E Class",
    "CL Class", "GLC Class", "GLA Class", "GLE Class"
]

transmission_options = ["Automatic", "Manual", "Semi-Auto"]
fuel_options = ["Petrol", "Diesel", "Hybrid"]

# ------------------------------
# Formulario de entrada
# ------------------------------
with st.form("car_form"):
    st.header("Introduce los datos del coche")

    model_car = st.selectbox("Modelo", model_options)
    year = st.number_input("Año", min_value=1990, max_value=2025, value=2020)

    transmission = st.selectbox("Transmisión", transmission_options)
    mileage = st.number_input("Kilometraje (km)", min_value=0, value=10000, step=1000)
    fuelType = st.selectbox("Combustible", fuel_options)

    engineSize = st.number_input(
        "Tamaño del motor (L)",
        min_value=0.5,
        value=2.0,
        step=0.1
    )

    tax = st.number_input(
        "Impuesto anual (€)",
        min_value=0,
        value=150,
        step=10
    )

    mpg = st.number_input(
        "Consumo (mpg)",
        min_value=10.0,
        value=50.0,
        step=1.0
    )

    submit = st.form_submit_button("Predecir precio")

# ------------------------------
# Predicción
# ------------------------------
if submit:
    try:
        input_data = pd.DataFrame({
            "model": [model_car],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuelType],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize]
        })

        # Limpieza defensiva (igual que en el EDA)
        input_data = input_data.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        with st.spinner("Calculando precio..."):
            precio = model.predict(input_data)[0]

        precio_fmt = f"{precio:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        st.success(f"💰 Precio estimado: €{precio_fmt}")

    except Exception as e:
        st.error("❌ Se produjo un error en la predicción")
        st.exception(e)
