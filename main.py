import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# URL for the background image
background_image_url = "https://blog.agribegri.com/public/blog_images/understanding-surface-irrigation-methods-600x400.webp"  # Replace with a valid direct image URL

# Custom CSS for background image and left text alignment with bold font
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    body, .left-text, h1, h2, h3, h4, h5, h6 {{
        text-align: left;  /* Align all text to the left */
        color: #ffffff;  /* Dark text color for readability */
        font-weight: bold;  /* Bold font */
        text-shadow: 1px 1px 2px #FFFFFF;  /* Optional shadow for better contrast */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display a left-aligned heading with bold text
st.markdown("<div class='left-text'><h1>**KRISHIASTRA - FARMING WITH PASSION ALWAYS**</h1></div>", unsafe_allow_html=True)

# Display left-aligned introductory text
st.markdown("<div class='left-text'>THE FARMER IS THE ONLY MAN IN OUR ECONOMY WHO BUYS EVERYTHING AT RETAIL, SELLS EVERYTHING AT WHOLESALE, AND PAYS THE FREIGHT BOTH WAYS.</div>", unsafe_allow_html=True)

# Load dataset with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return None

# File upload option
uploaded_file = st.file_uploader("Upload your crop dataset CSV file", type="csv")
crop = load_data(uploaded_file)

if crop is not None:
    st.markdown("<div class='left-text'>Dataset loaded successfully! You can proceed with analyzing the data and making predictions.</div>", unsafe_allow_html=True)

    # Define required columns
    required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_column = 'label'

    # Check if the necessary columns are present in the dataset
    missing_columns = [col for col in required_features if col not in crop.columns]
    if target_column not in crop.columns:
        missing_columns.append(target_column)

    if missing_columns:
        st.error(f"The following required columns are missing from the dataset: {missing_columns}")
    else:
        # Select only the required columns
        X_classification = crop[required_features]
        y_classification = crop[target_column]

        # Split data
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_classification, y_classification, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_cls_scaled = scaler.fit_transform(X_train_cls)
        X_test_cls_scaled = scaler.transform(X_test_cls)

        # Train a RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train_cls_scaled, y_train_cls)

        # Function to recommend a crop based on input features
        def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_data_scaled = scaler.transform(input_data)
            crop_prediction = classifier.predict(input_data_scaled)[0]
            return crop_prediction

        # User options
        st.markdown("<div class='left-text'><h2>Options</h2></div>", unsafe_allow_html=True)
        choice = st.radio("Choose an option:", ["Recommend a crop"])

        if choice == "Recommend a crop":
            # Input field parameters
            N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
            P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
            K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
            temperature = st.number_input("Temperature", min_value=0.0, step=1.0)
            humidity = st.number_input("Humidity", min_value=0.0, step=1.0)
            ph = st.number_input("pH Level", min_value=0.0, step=0.1)
            rainfall = st.number_input("Rainfall", min_value=0.0, step=1.0)

            if st.button("Recommend Crop"):
                recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
                st.markdown(f"<div class='left-text'>The Recommended Crop for the given conditions is: **{recommended_crop.capitalize()}**</div>", unsafe_allow_html=True)
