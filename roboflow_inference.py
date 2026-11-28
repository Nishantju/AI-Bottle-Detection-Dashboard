import os

from dotenv import load_dotenv
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID")

# Validate environment variables
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY missing from .env")

if not WORKSPACE:
    raise ValueError("ROBOFLOW_WORKSPACE missing from .env")

if not WORKFLOW_ID:
    raise ValueError("ROBOFLOW_WORKFLOW_ID missing from .env")


# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)


def predict_image(image_path):

    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={
            "image": image_path
        },
        use_cache=True
    )

    return result