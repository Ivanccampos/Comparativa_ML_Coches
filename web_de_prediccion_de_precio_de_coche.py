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

import base64

def add_local_bg(image_file):
    # Abrir y codificar la imagen local
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        /* 1. Imagen de fondo total */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}

        /* 2. Ajuste del contenedor principal */
        .main .block-container {{
            padding-top: 5rem;
            max-width: 600px;
        }}

        /* 3. CUADRO DE INPUTS CON OPACIDAD AJUSTABLE */
        [data-testid="stForm"] {{
            background-color: rgba(255, 255, 255, 0.4) !important;
            
            /* Efecto de desenfoque opcional */
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        /* 4. Estilo del resultado */
        .stAlert {{
            background-color: #00c853 !important;
            color: white !important;
        }}

        /* Título */
        h1 {{
            color: white !important;
            text-shadow: 2px 2px 10px #000000;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_local_bg("wp1828719-mercedes-benz-wallpapers.png")

st.title("Predicción de Precio de Coche")

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
    year = st.slider(
        "Año del coche", 
        min_value=2001, 
        max_value=2020, 
        value=2015  # Valor por defecto donde aparecerá el selector al cargar
    )
    transmission = st.selectbox("Transmisión", transmission_options)
    mileage = st.number_input("Kilometraje (km)", min_value=0, value=10000, step=1000)
    fuelType = st.selectbox("Combustible", fuel_options)

    engineSize = st.slider(
        "Tamaño del motor (L)",
        min_value=0,
        max_value=6.2,
        value=2
    )

    tax = st.slider(
        "Impuesto anual (€)",
        min_value=0,
        max_value=580,
        value=150
    )

    mpg = st.slider(
        "Consumo (mpg)",
        min_value=1.1,
        max_value=80.7,
        value=20
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
