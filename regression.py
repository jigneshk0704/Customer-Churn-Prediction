import streamlit as st
import numpy as np
import onnxruntime as rt
import pandas as pd
import pickle

st.set_page_config(page_title="Salary Prediction", page_icon="💰", layout="centered")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div.stButton > button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.85; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <span style='font-size:48px;'>💰</span>
        <h1 style='margin:0.3rem 0 0; font-size:2rem;'>Salary Prediction</h1>
        <p style='color:gray; font-size:15px; margin-top:0.3rem;'>
            Enter customer details to predict estimated salary
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Load model & encoders
session = rt.InferenceSession('regression_model.onnx')
input_name = session.get_inputs()[0].name

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.markdown("### 📋 Customer Information")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    age = st.slider('🎂 Age', 18, 92, 35)
    balance = st.number_input('💰 Balance', min_value=0.0, step=100.0)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col2:
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    credit_score = st.number_input('📊 Credit Score', min_value=0.0, step=1.0)
    tenure = st.slider('📅 Tenure (years)', 0, 10, 3)
    is_active_member = st.selectbox('🟢 Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    exited = st.selectbox('🚪 Exited', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

st.divider()

if st.button('💰 Predict Salary'):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_data_scaled = scaler.transform(input_data).astype(np.float32)

    prediction = session.run(None, {input_name: input_data_scaled})
    salary = prediction[0][0][0]

    st.markdown("### 🎯 Result")
    st.metric("Predicted Salary", f"${salary:,.2f}")
    st.success(f"💵 The estimated salary for this customer is **${salary:,.2f}**")
