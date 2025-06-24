import streamlit as st
import numpy as np 
import pandas as pd
from joblib import load

### loading saved models

linear_model = load('linear_regression.pkl')
ridge_model = load('ridge.pkl')
elastic_model = load('elastic.pkl')
lasso_model = load('lasso.pkl')
scaled_model = load('scaler.pkl')

df = pd.read_csv('50_Startups_dataset.csv')

feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit',
       'State_California', 'State_Florida', 'State_New_York']

st.title('Startup Profit Prediction App')
st.header('Enter Startup Details: ')

r_d_spend = st.number_input('R&D Spend', min_value=0, step=1000)
admin_spend = st.number_input("Administration Spend", min_value=0, step=1000)
marketing_spend = st.number_input("Marketing Spend", min_value=0, step=1000)
state = st.selectbox('State', ['California', 'Florida', 'New York'])

state_encoded = [0,0,0]
if state == 'California':
    state_encoded[0] = 1
elif state == 'Florida':
    state_encoded[1] = 1
elif state == 'New York':
    state_encoded[2] = 1

input_data = np.array([r_d_spend, admin_spend, marketing_spend] + state_encoded).reshape(1,-1)
input_data_scaled = scaled_model.transform(input_data)


# Predictions
if st.button("Predict Profit"):
    linear_pred = linear_model.predict(input_data_scaled)[0]
    ridge_pred = ridge_model.predict(input_data_scaled)[0]
    lasso_pred = lasso_model.predict(input_data_scaled)[0]
    elasticnet_pred = elastic_model.predict(input_data_scaled)[0]

    st.subheader("Predicted Profits:")
    st.write(f"Linear Regression: ${linear_pred:.2f}")
    st.write(f"Ridge Regression: ${ridge_pred:.2f}")
    st.write(f"Lasso Regression: ${lasso_pred:.2f}")
    st.write(f"ElasticNet Regression: ${elasticnet_pred:.2f}")

   