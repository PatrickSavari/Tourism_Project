import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import numpy as np
from huggingface_hub import HfApi
import joblib
import os
import shutil
import pandas as pd
from datasets import load_dataset
# Only import userdata if running in Colab to avoid errors in other environments
try:
    from google.colab import userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# Get Hugging Face token securely
if IS_COLAB:
    HF_TOKEN = userdata.get('HF_TOKEN')
else:
    HF_TOKEN = os.environ.get('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Please set it in Colab secrets or as an environment variable.")

# Define Hugging Face variables (consistent with data_prep.py and data_register.py)
HUGGINGFACE_USERNAME = "PatrickSavari"
DATASET_REPO_NAME = "tourism-customer-data"
TARGET_HF_REPO_ID = f"{HUGGINGFACE_USERNAME}/{DATASET_REPO_NAME}"

# Define model repo variables
HF_MODEL_REPO_NAME = "XGBoostTourismClassifier"
HF_MODEL_REPO_ID = f"{HUGGINGFACE_USERNAME}/{HF_MODEL_REPO_NAME}"

# Load train and test data from Hugging Face
print(f"Loading data from Hugging Face dataset: {TARGET_HF_REPO_ID}")
dataset_hf = load_dataset(TARGET_HF_REPO_ID, token=HF_TOKEN)
train_df = dataset_hf['train'].to_pandas()
test_df = dataset_hf['test'].to_pandas()

X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Set MLflow tracking URI (from the variable defined in the notebook)
# This allows switching between local and remote tracking
if 'MLFLOW_TRACKING_URI' in os.environ:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
elif 'MLFLOW_TRACKING_URI' in globals():
    mlflow.set_tracking_uri(globals()['MLFLOW_TRACKING_URI'])
else:
    mlflow.set_tracking_uri("file:./mlruns") # Default to local if not set

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Enable autologging for scikit-learn models (disabled for fine-grained control)
mlflow.sklearn.autolog(disable=True)

# Define MLflow experiment name
EXPERIMENT_NAME = "Tourism_Product_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="XGBoost_Training_Run") as run:
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("stratify_target", "ProdTaken")

    print("Starting preprocessing...")
    # Preprocessing: Label Encoding for categorical features and StandardScaler for numerical
    # Identify categorical and numerical columns (excluding the target and 'Unnamed: 0')
    categorical_cols = X_train.select_dtypes(include='object').columns
    numerical_cols = X_train.select_dtypes(include=np.number).columns.drop(['Unnamed: 0'], errors='ignore')


    # Apply Label Encoding
    le_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on training data
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
        # Transform test data
        X_test[col] = le.transform(X_test[col])
        le_encoders[col] = le # Store encoder

    # Apply StandardScaler to numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("Preprocessing complete.")

    # Define a temporary local directory to build the overall MLflow artifact structure
    temp_mlflow_artifact_root = "temp_mlflow_root_artifact_build"
    os.makedirs(temp_mlflow_artifact_root, exist_ok=True)

    # Define the subdirectory name within the MLflow artifacts that will hold the model
    model_artifact_subdir_name = "xgboost_model"

    # This is the path where mlflow.sklearn.save_model will save its files.
    # It will create this directory, ensuring it's empty when it writes.
    mlflow_core_model_path = os.path.join(temp_mlflow_artifact_root, model_artifact_subdir_name)

    # Model Training (XGBoost Classifier)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Model Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) # Corrected: Passed y_pred instead of f1_score function

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save the trained model using mlflow.sklearn.save_model
    # This will create mlflow_core_model_path and populate it with MLmodel, model.pkl, etc.
    mlflow.sklearn.save_model(
        sk_model=model,
        path=mlflow_core_model_path,
    )

    # Now, save preprocessors *inside* the MLflow model's directory
    preprocessors_save_path = os.path.join(mlflow_core_model_path, "preprocessors")
    os.makedirs(preprocessors_save_path, exist_ok=True) # Create the preprocessors subdirectory
    joblib.dump(le_encoders, os.path.join(preprocessors_save_path, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(preprocessors_save_path, "scaler.pkl"))

    # Log the *entire* temp_mlflow_artifact_root as an MLflow artifact
    # with an empty artifact_path. This means its contents will be logged directly
    # at the root of the MLflow run's artifact storage.
    # So, the 'xgboost_model' folder (with model and preprocessors) will appear
    # directly under 'artifacts/' in MLflow UI.
    mlflow.log_artifacts(
        local_dir=temp_mlflow_artifact_root,
        artifact_path="",
    )

    # Register the model. The model_uri should point to the subdirectory within the logged artifacts.
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/{model_artifact_subdir_name}", # This path should exist within the logged artifacts
        name="XGBoostTourismClassifier"
    )

    print("Model training, evaluation, and logging to MLflow complete.")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"MLflow Tracking UI: {mlflow.get_tracking_uri()}")

    # Register the best model in the Hugging Face model hub
    api_hf = HfApi(token=HF_TOKEN) # Initialize HfApi here, inside the run context

    try:
        api_hf.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", token=HF_TOKEN, private=False)
        print(f"Model repository '{HF_MODEL_REPO_NAME}' created successfully on Hugging Face.")
    except Exception as e:
        if "You already have a model named" in str(e):
            print(f"Model repository '{HF_MODEL_REPO_NAME}' already exists on Hugging Face.")
        else:
            print(f"Error creating repository: {e}")

    # Upload the MLflow model artifacts to Hugging Face
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='XGBoostTourismClassifier'")
    latest_version = max([mv.version for mv in model_versions])

    # Define a temporary local directory to download the model artifacts for HF upload
    temp_download_path = "hf_model_upload_temp"
    if os.path.exists(temp_download_path):
        shutil.rmtree(temp_download_path)
    os.makedirs(temp_download_path, exist_ok=True)

    # Download the entire 'xgboost_model' artifact (which now includes preprocessors)
    local_downloaded_model_path = client.download_artifacts(
        run_id=run.info.run_id,
        path=model_artifact_subdir_name, # "xgboost_model" artifact
        dst_path=temp_download_path
    )
    print(f"MLflow model artifacts downloaded locally to: {local_downloaded_model_path}")

    # Upload the contents of 'local_downloaded_model_path' to Hugging Face.
    # This will create a structure like HF_MODEL_REPO_ID/MLmodel, HF_MODEL_REPO_ID/model.pkl, HF_MODEL_REPO_ID/preprocessors/...
    api_hf.upload_folder(
        folder_path=local_downloaded_model_path,
        repo_id=HF_MODEL_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
        commit_message=f"Add XGBoostTourismClassifier v{latest_version}"
    )

    print(f"Model 'XGBoostTourismClassifier' (version {latest_version}) uploaded to Hugging Face: https://huggingface.co/models/{HF_MODEL_REPO_ID}")

    # Clean up the temporary local MLflow artifact build directory and download directory
    shutil.rmtree(temp_mlflow_artifact_root)
    shutil.rmtree(temp_download_path)
    print(f"Cleaned up temporary directories.")
