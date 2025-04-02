import streamlit as st
import pickle
import numpy as np

# Load model
data = pickle.load(open('bhp_model.pkl', 'rb'))
lr_clf = data['model']
columns = data['columns']


# Streamlit UI components
st.title("🏡 Bangalore House Price Prediction")
st.write("Enter details below to estimate the house price.")

# User Inputs
sqft = st.number_input("Enter area in square feet", min_value=1000, step=100)
bhk = st.number_input("Enter number of bedrooms", min_value=1, step=1)
bath = st.number_input("Enter number of bathrooms", min_value=1, step=1)
location = st.selectbox("Select location", columns[3:])

# Price Prediction Function
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1
    
    return lr_clf.predict([x])[0]   # Convert to Lakhs

# Estimate Button
if st.button("Estimate Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"🏠 Estimated House Price: ₹{price:,.2f} Lakhs")