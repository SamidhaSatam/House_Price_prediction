import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD MODEL ----------------
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ---------------- UI ----------------
st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the price")

# ---- NUMERIC INPUTS ----
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5)
sqft_living = st.number_input("Living Area (sqft)", min_value=0)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=0)
floors = st.number_input("Floors", min_value=0.0, step=0.5)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
sqft_above = st.number_input("Sqft Above", min_value=0)
sqft_basement = st.number_input("Sqft Basement", min_value=0)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2025)
yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0)

# ---- STATEZIP INPUT ----
statezip_options = [col.replace("statezip_", "") 
                    for col in model_columns if col.startswith("statezip_")]

statezip = st.selectbox("State Zip", sorted(statezip_options))

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    # Create zero-filled input
    input_data = np.zeros(len(model_columns))

    # Fill numeric values
    input_dict = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated
    }

    for i, col in enumerate(model_columns):
        if col in input_dict:
            input_data[i] = input_dict[col]
        elif col == f"statezip_{statezip}":
            input_data[i] = 1

    # Predict
    prediction = model.predict(input_data.reshape(1, -1))

    st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")

import joblib

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")


