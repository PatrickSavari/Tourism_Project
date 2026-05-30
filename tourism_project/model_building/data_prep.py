import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi
import os
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

# Define Hugging Face username and repository name (from previous cells)
HUGGINGFACE_USERNAME = "PatrickSavari" # Replace with your Hugging Face username
DATASET_REPO_NAME = "tourism-customer-data"
TARGET_HF_REPO_ID = f"{HUGGINGFACE_USERNAME}/{DATASET_REPO_NAME}"

# Load the dataset directly from Hugging Face Hub
# The dataset was uploaded as 'tourism.csv' in the root of the dataset repo
print(f"Loading dataset from Hugging Face: {TARGET_HF_REPO_ID}")
dataset = load_dataset(TARGET_HF_REPO_ID, data_files={'train': 'tourism.csv'}, token=HF_TOKEN)

# Access the 'train' split of the dataset and convert to pandas DataFrame
df = dataset['train'].to_pandas()

print("Dataset loaded successfully:")
print(f"Shape of the dataset: {df.shape}")

# Data Cleaning

# Drop 'CustomerID' as it's an identifier and not useful for modeling
# Also drop 'Designation' and 'MonthlyIncome' since they have many missing values and might not be directly relevant or are highly correlated with other features
# Revisit this decision if model performance is not satisfactory
df = df.drop(columns=['CustomerID', 'Designation', 'MonthlyIncome'], errors='ignore')

# Handle missing values: Fill numerical NaNs with the mean and categorical NaNs with the mode.
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            # For categorical data, fill with the mode, but handle cases where mode might be empty
            mode_val = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
            df[column] = df[column].fillna(mode_val)

print("Dataset after cleaning and handling missing values:")
print(f"Shape of the cleaned dataset: {df.shape}")
print("Missing values after cleaning:")
print(df.isnull().sum())

# Data Splitting

# Define features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Combine X and y for train and test sets for easier saving
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

#Save and Upload to HF

# Define paths to save locally
LOCAL_DATA_DIR = "tourism_project/data"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
TRAIN_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "test.csv")

# Save the split datasets locally
train_df.to_csv(TRAIN_FILE_PATH, index=False)
test_df.to_csv(TEST_FILE_PATH, index=False)

print(f"Training data saved locally to: {TRAIN_FILE_PATH}")
print(f"Testing data saved locally to: {TEST_FILE_PATH}")

api = HfApi(token=HF_TOKEN)

# Upload train.csv to Hugging Face Hub
api.upload_file(
    path_or_fileobj=TRAIN_FILE_PATH,
    path_in_repo="train.csv", # Name in HF repo
    repo_id=TARGET_HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Add train.csv dataset"
)
print("----------------------------------------------------------------------------------------------------------")
print(f"'train.csv' uploaded to Hugging Face dataset: https://huggingface.co/datasets/{TARGET_HF_REPO_ID}/blob/main/train.csv")
print("----------------------------------------------------------------------------------------------------------")

# Upload test.csv to Hugging Face Hub
api.upload_file(
    path_or_fileobj=TEST_FILE_PATH,
    path_in_repo="test.csv", # Name in HF repo
    repo_id=TARGET_HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Add test.csv dataset"
)
print("----------------------------------------------------------------------------------------------------------")
print(f"'test.csv' uploaded to Hugging Face dataset: https://huggingface.co/datasets/{TARGET_HF_REPO_ID}/blob/main/test.csv")
print("----------------------------------------------------------------------------------------------------------")
