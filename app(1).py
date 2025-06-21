
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Fertilizer Quality Checker", layout="centered")

st.title("🧪 Fertilizer Quality Checker - AI-Powered")
st.write("This app helps farmers detect fake or unsuitable fertilizers before using them in the field.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Fertilizer Data (.csv)", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        st.dataframe(data.head())

        if "Status" in data.columns:
            # Simple model training (for demo purposes)
            st.info("Training model to detect fake fertilizers...")
            X = data.drop("Status", axis=1)
            y = data["Status"]

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            st.success("Model trained! You can now test new fertilizer entries.")

            # Prediction section
            st.subheader("🧠 Test a New Fertilizer Entry")
            input_data = {}
            for column in X.columns:
                value = st.number_input(f"{column}", value=0.0)
                input_data[column] = value

            if st.button("Check Fertilizer"):
                user_df = pd.DataFrame([input_data])
                prediction = model.predict(user_df)[0]
                if prediction == "Fake":
                    st.error("⚠️ Warning: Fertilizer may be fake or unsuitable!")
                else:
                    st.success("✅ This fertilizer seems safe to use.")
        else:
            st.warning("Dataset must contain a 'Status' column with labels like 'Real' or 'Fake'.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a fertilizer dataset CSV to begin.")

st.markdown("---")
st.caption("Made by Bryan with ❤️ and Streamlit")
