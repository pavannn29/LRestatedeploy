import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import requests
from io import BytesIO

# model_path = '/Users/pavan/MLProjects/MLClass/linear_regression_model.joblib'
#model_path = 'https://raw.githubusercontent.com/pavannn29/LRestatedeploy/main/data/linear_regression_model.joblib'
model_path = 'https://drive.google.com/file/d/1taQGa3kikK4IHGM_G9cfMAm0cJ824dho/view?usp=sharing'
#csv_path = '/Users/pavan/MLProjects/MLClass/LRestate.csv'
#csv_path = 'https://raw.githubusercontent.com/pavannn29/LRestatedeploy/main/data/LRestate.csv'
csv_path = 'https://drive.google.com/file/d/1nGzG2-q5DZ2WFPF_8Cq4urcBE7_vCpmn/view?usp=sharing'
# Load the trained model
# model = joblib.load(model_path)

# Function to fetch data and load the model
def fetch_data_and_model():
    # Fetch the data file
    data_content = requests.get(csv_path).content
    data = pd.read_csv(BytesIO(data_content))

    # Fetch the model file
    model_content = requests.get(model_path).content
    model = joblib.loads(model_content)

    return data, model

# Load data and model during app initialization
data, model = fetch_data_and_model()

# Load the label encoder and scaler used during training
label_encoder = LabelEncoder()
scaler = MinMaxScaler()
data = pd.read_csv(csv_path)
label_encoder.fit(data["location"])
scaler.fit(data.drop(["price", "location"], axis=1))

# Function to predict the price
def predict_price(area, bedrooms, bathrooms, location):
    # Encode the location
    location_encoded = label_encoder.transform([location])[0]

    # Scale the input features
    input_features = np.array([[area, bedrooms, bathrooms]])  # Exclude "location_encoded"
    input_features_scaled = scaler.transform(input_features)

    # Add the location_encoded feature back to the scaled input_features
    input_features_scaled = np.hstack((input_features_scaled, [[location_encoded]]))

    # Make the prediction using the trained model
    predicted_price = model.predict(input_features_scaled)[0]
    return predicted_price

# Streamlit app
def main():
    st.title("Real Estate Price Prediction")
    
    # User input
    area = st.number_input("Area (sq. ft.)", min_value=0)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0)
    location = st.selectbox("Location", ["New York", "Los Angeles", "Chicago"])

    # Predict price button
    if st.button("Predict Price"):
        predicted_price = predict_price(area, bedrooms, bathrooms, location)
        st.success(f"Predicted Price: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
