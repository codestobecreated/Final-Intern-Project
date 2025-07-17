import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os

# -----------------------------
# STEP 1: LOAD & CLEAN DATASET
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("cleaned_amazon_sales.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    required_columns = ['category', 'fulfilment', 'status', 'sales_channel', 'ship-state', 'currency', 'amount']
    df = df[required_columns].dropna()

    return df

df = load_and_prepare_data()

# -----------------------------
# STEP 2: TRAIN AND SAVE MODEL (IF NOT EXISTS)
# -----------------------------
model_path = "mlp_model.pkl"
scaler_path = "scaler.pkl"
encoder_path = "encoder.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
    X = df.drop("amount", axis=1)
    y = df["amount"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

# -----------------------------
# STEP 3: STREAMLIT UI
# -----------------------------
st.title("📦 Amazon Sales Amount Predictor")
st.subheader("Enter Product & Shipping Info to Predict Amount")

category = st.selectbox("Category", sorted(df["category"].dropna().unique()))
fulfilment = st.selectbox("Fulfilment", sorted(df["fulfilment"].dropna().unique()))
status = st.selectbox("Order Status", sorted(df["status"].dropna().unique()))
sales_channel = st.selectbox("Sales Channel", sorted(df["sales_channel"].dropna().unique()))
ship_state = st.selectbox("Shipping State", sorted(df["ship-state"].dropna().unique()))
currency = st.selectbox("Currency", sorted(df["currency"].dropna().unique()))

input_data = {
    'category': [category],
    'fulfilment': [fulfilment],
    'status': [status],
    'sales_channel': [sales_channel],
    'ship-state': [ship_state],
    'currency': [currency]
}
input_df = pd.DataFrame(input_data)

encoded_input = encoder.transform(input_df)
scaled_input = scaler.transform(encoded_input)

if st.button("Predict Sales Amount 💸"):
    prediction = model.predict(scaled_input)
    st.success(f"💰 Estimated Sales Amount: ₹{prediction[0]:,.2f}")
