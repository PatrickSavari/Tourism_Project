import streamlit as st
import pandas as pd
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import joblib
from huggingface_hub import HfApi, snapshot_download # Import HfApi and snapshot_download

# # Only import userdata if running in Colab to avoid errors in other environments
# try:
#     from google.colab import userdata
#     IS_COLAB = True
# except ImportError:
#     IS_COLAB = False

# # Get Hugging Face token securely
# if IS_COLAB:
#     HF_TOKEN = userdata.get('HF_TOKEN')
# else:
#     HF_TOKEN = os.environ.get('HF_TOKEN')

# if not HF_TOKEN:
#     # This might happen if running locally without the token set.
#     # For a Streamlit app deployed on HF Spaces, the HF_TOKEN is usually
#     # automatically available as an environment variable, so this check
#     # is mostly for local development.
#     st.warning("HF_TOKEN not found. Model download might fail if the repo is private.")
#     HF_TOKEN = None # Set to None if not found

# Define model repo variables (consistent with model_training.py)
HUGGINGFACE_USERNAME = "PatrickSavari"
HF_MODEL_REPO_NAME = "XGBoostTourismClassifier"
HF_MODEL_REPO_ID = f"{HUGGINGFACE_USERNAME}/{HF_MODEL_REPO_NAME}"

# Define a local directory to download the model and preprocessors within the Space
LOCAL_MODEL_DOWNLOAD_PATH = "./hf_downloaded_model"

# Use snapshot_download to get the model artifacts from the Hugging Face Model Hub
try:
    snapshot_download(
        repo_id=HF_MODEL_REPO_ID,
        local_dir=LOCAL_MODEL_DOWNLOAD_PATH,
        #token=HF_TOKEN # Use the securely obtained token
    )
    st.success(f"Model and preprocessors downloaded from Hugging Face Model Hub to {LOCAL_MODEL_DOWNLOAD_PATH}")
except Exception as e:
    st.error(f"Error downloading model from Hugging Face Model Hub: {e}")
    st.stop() # Stop the app if download fails

# The MLflow model artifacts are at the root of LOCAL_MODEL_DOWNLOAD_PATH after snapshot_download
model_path = LOCAL_MODEL_DOWNLOAD_PATH

# Load the MLflow pyfunc model
model = mlflow.pyfunc.load_model(model_path)

# Load the saved preprocessors from the downloaded path
# They are within LOCAL_MODEL_DOWNLOAD_PATH/preprocessors folder
le_encoders_path = os.path.join(LOCAL_MODEL_DOWNLOAD_PATH, "preprocessors", "label_encoders.pkl")
scaler_path = os.path.join(LOCAL_MODEL_DOWNLOAD_PATH, "preprocessors", "scaler.pkl")

try:
    le_encoders = joblib.load(le_encoders_path)
    scaler = joblib.load(scaler_path)
    st.success("Preprocessors loaded successfully.")
except Exception as e:
    st.error(f"Error loading preprocessors: {e}")
    st.stop() # Stop the app if preprocessors fail to load

st.set_page_config(layout="wide") # Set layout to wide to utilize full screen width
st.title("Tourism Product Purchase Prediction")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for customer details arranged in columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
    number_of_person_visiting = st.slider("Number Of Person Visiting", 1, 5, 2)
    number_of_children_visiting = st.slider("Number Of Children Visiting", 0, 5, 0)

with col2:
    duration_of_pitch = st.slider("Duration Of Pitch (minutes)", 0, 60, 10)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male']) # 'Fe Male' is a common typo in datasets
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    number_of_trips = st.slider("NumberOfTrips (annually)", 0, 20, 5)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
    number_of_followups = st.slider("NumberOfFollowups", 0, 10, 3)

# Binary inputs often look good on their own or grouped at the bottom
col3, col4 = st.columns(2)
with col3:
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
with col4:
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')


if st.button("Predict Purchase"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Unnamed: 0': 0, # Placeholder, assuming it's an identifier or dropped. Keep for column consistency.
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
    }])

    # Preprocessing: Apply the loaded LabelEncoders and StandardScaler
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
    numerical_cols = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                      'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                      'NumberOfChildrenVisiting']

    # Apply Label Encoding using loaded encoders
    for col in categorical_cols:
        try:
            input_data[col] = le_encoders[col].transform(input_data[col])
        except ValueError as e:
            st.warning(f"Warning: Category not seen during training for {col}. Error: {e}")
            input_data[col] = -1 # Assign a default for unseen categories

    # Apply StandardScaler to numerical features using the loaded scaler
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Ensure column order matches training data's X_train
    final_columns = ['Unnamed: 0', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender',
                     'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
                     'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                     'NumberOfChildrenVisiting']
    processed_input_df = input_data[final_columns]

    try:
        prediction = model.predict(processed_input_df)
        # # Attempt to get probability from the pyfunc model first
        # try:
        #     probability = model.predict_proba(processed_input_df)[:, 1]
        # except AttributeError:
        #     # If predict_proba is not directly available, load the raw sklearn model for it
        #     st.warning("mlflow.pyfunc model does not have 'predict_proba'. Attempting to load raw sklearn model.")
        #     raw_sklearn_model_path = os.path.join(model_path, "model.pkl")
        #     if os.path.exists(raw_sklearn_model_path):
        #         raw_model = joblib.load(raw_sklearn_model_path)
        #         probability = raw_model.predict_proba(processed_input_df)[:, 1]
        #     else:
        #         st.error(f"Failed to find raw sklearn model at {raw_sklearn_model_path}.")
        #         probability = np.array([0.5]) # Default to 0.5 if probability cannot be determined

        if prediction[0] == 1:
            st.success(f"Prediction: This customer is LIKELY to purchase the package")
        else:
            st.info(f"Prediction: This customer is UNLIKELY to purchase the package")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the model and preprocessing steps are correctly aligned.")
