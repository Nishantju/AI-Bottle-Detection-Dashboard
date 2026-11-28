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

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY missing from .env")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="V1bwqugzfPo3ygUXYDHL"
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