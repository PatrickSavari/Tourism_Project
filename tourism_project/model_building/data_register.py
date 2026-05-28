import os
from huggingface_hub import HfApi

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

# Define your Hugging Face username and repository name
HUGGINGFACE_USERNAME = "PatrickSavari" # Replace with your Hugging Face username
DATASET_REPO_NAME = "tourism-customer-data" # You can choose any name for your dataset repository

# Initialize Hugging Face API
api = HfApi(token=HF_TOKEN)

# Create the dataset repository on Hugging Face if it doesn't exist
try:
    api.create_repo(repo_id=f"{HUGGINGFACE_USERNAME}/{DATASET_REPO_NAME}", repo_type="dataset", token=HF_TOKEN, private=False)
    print(f"Dataset repository '{DATASET_REPO_NAME}' created successfully on Hugging Face.")
except Exception as e:
    if "You already have a dataset named" in str(e):
        print(f"Dataset repository '{DATASET_REPO_NAME}' already exists on Hugging Face.")
    else:
        print(f"Error creating repository: {e}")

# Define source file path and target path in Hugging Face repository
SOURCE_FILE_PATH = "tourism_project/data/tourism.csv"
TARGET_HF_REPO_ID = f"{HUGGINGFACE_USERNAME}/{DATASET_REPO_NAME}"
PATH_IN_HF_REPO = "tourism.csv" # The name of the file in the Hugging Face repository

# Upload the file using HfApi
api.upload_file(
    path_or_fileobj=SOURCE_FILE_PATH,
    path_in_repo=PATH_IN_HF_REPO,
    repo_id=TARGET_HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Add tourism.csv dataset"
)

print(f"'{PATH_IN_HF_REPO}' uploaded to Hugging Face dataset: https://huggingface.co/datasets/{TARGET_HF_REPO_ID}")
