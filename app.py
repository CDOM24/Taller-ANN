import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# -----------------------------
# Configuración página
# -----------------------------
st.set_page_config(
    page_title="Predicción Riesgo Crediticio",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Cargar modelo y scaler
# -----------------------------
model = load_model("modelo_credit_score.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Dataset solo para rangos
# -----------------------------
url = "https://raw.githubusercontent.com/CDOM24/Taller-ANN/main/riesgo.csv"
df = pd.read_csv(url, sep=";", decimal=",")

# -----------------------------
# Título
# -----------------------------
st.title("💳 Predicción de Riesgo Crediticio")

st.markdown(
"""
Esta aplicación usa **Inteligencia Artificial** para estimar  
el **nivel de riesgo crediticio de un cliente**.

Ingrese los datos financieros del cliente.
"""
)

st.divider()

# -----------------------------
# Entradas en columnas
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Edad",
        int(df["Age"].min()),
        int(df["Age"].max()),
        int(df["Age"].mean())
    )

    income = st.slider(
        "Ingreso anual",
        int(df["Annual_Income"].min()),
        int(df["Annual_Income"].max()),
        int(df["Annual_Income"].mean())
    )

    accounts = st.slider(
        "Número de cuentas bancarias",
        int(df["Num_Bank_Accounts"].min()),
        int(df["Num_Bank_Accounts"].max()),
        3
    )

with col2:

    cards = st.slider(
        "Número de tarjetas de crédito",
        int(df["Num_Credit_Card"].min()),
        int(df["Num_Credit_Card"].max()),
        2
    )

    debt = st.slider(
        "Deuda pendiente",
        float(df["Outstanding_Debt"].min()),
        float(df["Outstanding_Debt"].max()),
        float(df["Outstanding_Debt"].mean())
    )

    balance = st.slider(
        "Balance mensual",
        float(df["Monthly_Balance"].min()),
        float(df["Monthly_Balance"].max()),
        float(df["Monthly_Balance"].mean())
    )

st.divider()

# -----------------------------
# Botón predicción
# -----------------------------
if st.button("🔎 Analizar Riesgo Crediticio"):

    datos = np.array([[
        age,
        income,
        accounts,
        cards,
        debt,
        balance
    ]])

    datos = scaler.transform(datos)

    pred = model.predict(datos)

    clase = np.argmax(pred)

    prob = pred[0]

    st.subheader("Resultado del análisis")

    # -----------------------------
    # Interpretación
    # -----------------------------
    if clase == 0:
        st.success("🟢 Riesgo Bajo")
    elif clase == 1:
        st.warning("🟡 Riesgo Medio")
    else:
        st.error("🔴 Riesgo Alto")

    st.divider()
