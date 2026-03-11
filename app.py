import streamlit as st
import pandas as pd

# cargar dataset desde GitHub
url = "https://raw.githubusercontent.com/CDOM24/Taller-ANN/main/riesgo.csv"

df = pd.read_csv(url, sep=";", decimal=",")

st.title("Predicción de Riesgo Crediticio")

st.write("Dataset utilizado para entrenar el modelo")
st.dataframe(df.head())

st.header("Ingrese datos del cliente")

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

st.subheader("Datos ingresados")

st.write({
    "Edad": age,
    "Ingreso anual": income,
    "Cuentas": accounts,
    "Tarjetas": cards,
    "Deuda": debt,
    "Balance": balance
})