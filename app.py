import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model safely using caching
@st.cache_resource
def load_model():
    # Replace 'model.pkl' with the exact name of your file if it is different
    with open("model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# 2. App Title and Description
st.title("🚀 Machine Learning Model Predictor")
st.write("Enter the required inputs below to get a prediction from the trained model.")

st.divider()

# 3. User Input Fields
# Update these fields to match the exact features your model was trained on
st.subheader("📋 Input Features")

# Example Input 1: Numeric input with a slider
feature_1 = st.slider("Feature 1 (e.g., Age)", min_value=0, max_value=100, value=25)

# Example Input 2: Continuous numeric input box
feature_2 = st.number_input("Feature 2 (e.g., Income)", min_value=0.0, max_value=200000.0, value=50000.0)

# Example Input 3: Categorical input converted to numeric representation
feature_3_selection = st.selectbox("Feature 3 (e.g., Location)", options=["Urban", "Suburban", "Rural"])
# Convert the selection to whatever format your model expects (e.g., 0, 1, 2)
feature_3 = 0 if feature_3_selection == "Urban" else (1 if feature_3_selection == "Suburban" else 2)

st.divider()

# 4. Prediction Logic
if st.button("🔮 Run Prediction", type="primary"):
    # Arrange inputs into the exact shape/order the model expects
    # For a standard 2D array input: [[feature_1, feature_2, feature_3]]
    input_data = np.array([[feature_1, feature_2, feature_3]])
    
    try:
        prediction = model.predict(input_data)
        
        # Display results
        st.subheader("✨ Prediction Result")
        st.success(f"The model predicted value is: **{prediction[0]}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Check that the number and order of input features match what the model expects.")
