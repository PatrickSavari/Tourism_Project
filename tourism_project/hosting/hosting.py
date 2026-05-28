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

# Define your Hugging Face username and the name for your Streamlit Space
HUGGINGFACE_USERNAME = "PatrickSavari" # Replace with your Hugging Face username
STREAMLIT_SPACE_NAME = "tourism-product-predictor-app" # Choose a name for your Streamlit Space

HF_SPACE_REPO_ID = f"{HUGGINGFACE_USERNAME}/{STREAMLIT_SPACE_NAME}"

api = HfApi(token=HF_TOKEN)

# Create the Hugging Face Space repository if it doesn't exist
try:
    api.create_repo(repo_id=HF_SPACE_REPO_ID, repo_type="space", space_sdk="docker", token=HF_TOKEN, private=False)
    print(f"Hugging Face Space '{STREAMLIT_SPACE_NAME}' created successfully.")
except Exception as e:
    if "You already have a space named" in str(e):
        print(f"Hugging Face Space '{STREAMLIT_SPACE_NAME}' already exists.")
    else:
        print(f"Error creating Hugging Face Space: {e}")

# Define the local folder containing the deployment files
LOCAL_DEPLOYMENT_FOLDER = "tourism_project/deployment"

# Upload the entire deployment folder to the Hugging Face Space
api.upload_folder(
    folder_path=LOCAL_DEPLOYMENT_FOLDER,
    repo_id=HF_SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN,
    commit_message="Add Streamlit app and Dockerfile for deployment"
)

print(f"Deployment files from '{LOCAL_DEPLOYMENT_FOLDER}' uploaded to Hugging Face Space: https://huggingface.co/spaces/{HF_SPACE_REPO_ID}")
