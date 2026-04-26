import streamlit as st
import pandas as pd 
import tensorflow as tf
import numpy as np 
import pickle

# Page config
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stSlider label, .stNumberInput label, .stSelectbox label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 🔥 Decorative Title
st.markdown("""
    <h1 style='text-align: center; color: #00C9A7;'>
        💡 AI Smart Prediction System
    </h1>
    <p style='text-align: center; font-size:18px; color:gray;'>
        Predict outcomes using your custom inputs 🚀
    </p>
""", unsafe_allow_html=True)

st.divider()

# Load model
model = tf.keras.models.load_model('regression_model.h5')

# Load scaler & encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)   

# 📌 Input Section
st.subheader("📊 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input('💳 Credit Score')
    age = st.slider('🎂 Age', 18, 92)
    tenure = st.slider('📅 Tenure', 0, 10)
    balance = st.number_input('💰 Balance')

with col2:
    estimated_salary = st.number_input('💵 Estimated Salary')
    num_of_products = st.slider('📦 Number of Products', 1, 4)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('🟢 Active Member', [0, 1])

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)

st.divider()

# 🚀 Predict Button
if st.button(" Predict Now"):

    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Fix columns (IMPORTANT)
    if 'Exited' in scaler.feature_names_in_:
        input_data['Exited'] = 0

    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    # 🎯 Output Styling
    st.success(f"💰 Prediction Result: {prediction[0][0]:,.2f}")
