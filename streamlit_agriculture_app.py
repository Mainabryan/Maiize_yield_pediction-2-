
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="🌾 Kenya Maize Yield Predictor", layout="centered")

# Load model and scaler
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# Add custom style
st.markdown("""
    <style>
        .main {
            background-color: #f4f8f3;
            color: #1e3932;
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌽 Kenya Maize Yield Predictor")
st.markdown("This app uses Ridge Regression to predict **maize yield** based on environmental and farming factors in Kenya 🇰🇪.")

# Sidebar inputs
st.sidebar.header("🧑‍🌾 Farmer Data Input")
rainfall = st.sidebar.slider("🌧️ Average Rainfall (mm)", 100, 1000, 500)
temperature = st.sidebar.slider("🌡️ Average Temperature (°C)", 10, 40, 25)
soil_quality = st.sidebar.slider("🧪 Soil Quality Score", 0, 1, 1)
fertilizer = st.sidebar.slider("🧴 Fertilizer Used (kg/acre)", 0, 200, 100)
prev_yield = st.sidebar.number_input("📊 Previous Season Yield (t/ha)", 0.0, 10.0, 2.5)
altitude = st.sidebar.slider("⛰️ Altitude (meters)", 500, 3000, 1500)
seed_type = st.sidebar.selectbox("🌱 Seed Type", ["Local", "Hybrid"])
farming_method = st.sidebar.selectbox("🚜 Farming Method", ["Manual", "Mechanized"])
season = st.sidebar.selectbox("☁️ Planting Season", ["Long Rains", "Short Rains"])

# Encoding categorical inputs
seed_type_hybrid = 1 if seed_type == "Hybrid" else 0
farming_method_mech = 1 if farming_method == "Mechanized" else 0
season_short = 1 if season == "Short Rains" else 0

# Model input
X_input = np.array([[rainfall, temperature, soil_quality, fertilizer, prev_yield, altitude,
                     seed_type_hybrid, farming_method_mech, season_short]])
X_scaled = scaler.transform(X_input)

# Prediction
if st.sidebar.button("🎯 Predict Maize Yield"):
    prediction = model.predict(X_scaled)[0]
    st.subheader("📈 Predicted Yield")
    st.success(f"{prediction:.2f} tonnes per hectare")

    if prediction < 2:
        st.warning("⚠️ Low yield. Consider switching to hybrid seeds or improving soil quality.")
    elif 2 <= prediction <= 4:
        st.info("🔄 Moderate yield. You can optimize further with better fertilizer balance.")
    else:
        st.success("🌟 Great yield! Keep up your sustainable practices.")

# Footer
st.markdown("---")
st.caption("Built by Bryan Waweru · Powered by Ridge Regression · Agriculture meets AI 🌍")
