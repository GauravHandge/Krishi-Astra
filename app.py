import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
     <style>
    .krishiAstra {{
        color: #0000FF;  /* Blue for KrishiAstra */
        font-weight: bold;
    }}
    .passion-text {{
        color: #008000;  /* Green for other text */
        font-weight: bold;
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

    # Create a dynamic mapping for crop labels (case insensitive)
    unique_crops = crop['label'].str.lower().unique()
    crop_dict = {crop: idx + 1 for idx, crop in enumerate(unique_crops)}
    crop['crop_num'] = crop['label'].str.lower().map(crop_dict)

    # Prepare data for the classification model
    X_classification = crop.drop(['crop_num', 'label'], axis=1)
    y_classification = crop['crop_num']

    # Split data
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_cls = scaler.fit_transform(X_train_cls)
    X_test_cls = scaler.transform(X_test_cls)

    # Train a RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_cls, y_train_cls)

    # Train regression models for optimal parameters prediction
    X_regression = crop[['crop_num']]
    regressors = {}
    for param in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        y = crop[param]
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y, test_size=0.2, random_state=42)
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train_reg, y_train_reg)
        regressors[param] = regressor

    # Function to get optimal levels for a given crop
    def get_optimal_levels(crop_name):
        crop_name = crop_name.strip().lower()
        crop_num = crop_dict.get(crop_name)
        if crop_num is None:
            return "Crop not found. Please check the input."

        input_data = np.array([[crop_num]])
        optimal_levels = {param: regressor.predict(input_data)[0] for param, regressor in regressors.items()}
        return optimal_levels

    # Function to compare field parameters with optimal levels
    def compare_parameters(desired_crop, field_params):
        optimal_levels = get_optimal_levels(desired_crop)
        if isinstance(optimal_levels, str):
            return optimal_levels

        comparison_results = {}
        for param, field_value in field_params.items():
            optimal_value = optimal_levels[param]
            if field_value < optimal_value * 0.9:
                comparison_results[param] = f"Increase (current: {field_value}, optimal: {optimal_value:.2f})"
            elif field_value > optimal_value * 1.1:
                comparison_results[param] = f"Decrease (current: {field_value}, optimal: {optimal_value:.2f})"
            else:
                comparison_results[param] = f"Optimal range (current: {field_value}, optimal: {optimal_value:.2f})"
        return comparison_results

    # User options
    st.markdown("<div class='left-text'><h2>Options</h2></div>", unsafe_allow_html=True)
    choice = st.radio("Choose an option:", ["Compare Field Fertilizer Parameters", "Fertilizer Recommendation"])

    if choice == "Compare Field Fertilizer Parameters":
        desired_crop = st.text_input("Enter the crop you want to grow:")

        if desired_crop:
            # Input field parameters for comparison
            N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
            P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
            K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
            temperature = st.number_input("Temperature", min_value=0.0, step=1.0)
            humidity = st.number_input("Humidity", min_value=0.0, step=1.0)
            ph = st.number_input("pH Level", min_value=0.0, step=0.1)
            rainfall = st.number_input("Rainfall", min_value=0.0, step=1.0)

            field_params = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature, 'humidity': humidity,
                'ph': ph, 'rainfall': rainfall
            }

            if st.button("Compare Parameters"):
                comparison = compare_parameters(desired_crop, field_params)
                if isinstance(comparison, str):
                    st.markdown(f"<div class='left-text'>{comparison}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='left-text'>Parameter comparison for {desired_crop.capitalize()}:</div>", unsafe_allow_html=True)
                    for param, result in comparison.items():
                        st.markdown(f"<div class='left-text'>{param}: {result}</div>", unsafe_allow_html=True)

    elif choice == "Fertilizer Recommendation":
        # Input the crop name to get optimal parameters
        crop_name = st.text_input("Enter the Crop name for Fertilizer Recommendation:")

        if crop_name:
            optimal_params = get_optimal_levels(crop_name)
            if isinstance(optimal_params, str):
                st.markdown(f"<div class='left-text'>{optimal_params}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='left-text'>Fertilizer for {crop_name.capitalize()}:</div>", unsafe_allow_html=True)
                for param, value in optimal_params.items():
                    st.markdown(f"<div class='left-text'>{param}: {value:.2f}</div>", unsafe_allow_html=True)
