from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import os
from PIL import Image
import traceback
import uvicorn
import requests
import json
import logging

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from fastapi.middleware.cors import CORSMiddleware
# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="app.log"
)
logger = logging.getLogger("fastapi-backend")

img_data_location = os.getenv("IMG_DATA_LOCATION", "./uploaded_img_data/category")
mlflow_hosted_model_url = os.getenv("MLFLOW_HOSTED_MODEL_URL", "http://localhost:5002/invocations/")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prometheus metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds')
save_image_requests = Counter('save_image_requests_total', 'Total number of image save requests')
save_image_latency = Histogram('save_image_latency_seconds', 'Image save latency in seconds')
error_counter = Counter('api_errors_total', 'Total number of API errors', ['endpoint', 'error_type'])

class ImageData(BaseModel):
    image: str  # Base64 encoded string
    category: str # string

class ImgDataForPrediction(BaseModel):
    inputs: str

def update_count_file():
    try:
        with open("count.txt", "wb") as f:
            count = int(f.read().strip())
            count += 1
            f.write(count)
    except Exception as e:
        logger.error(f"Failed to update count file: {e}")

@app.post("/save-image/")
async def save_image(data: ImageData):
    save_image_requests.inc()
    start_time = time.time()
    logger.info(f"Received request to save image in category '{data.category}'")
    try:
        # Decode the base64 string
        image_bytes = base64.b64decode(data.image)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        logger.info(f"Decoded image array length: {len(image_array)}")
        print(len(image_array))

        if len(image_array) != 784:
            logger.warning("Image does not have 784 values (28x28 pixels)")
            raise HTTPException(status_code=400, detail="Image must be 28x28 pixels (784 values)")

        image = image_array.reshape(28, 28)
        parent_path = img_data_location

        path = os.path.join(parent_path, data.category)
        logger.info(f"Saving image to path: {path}")
        os.makedirs(path, exist_ok=True)
        files_only = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        file_name = str(len(files_only)) + ".png"

        img = Image.fromarray(image)
        img.save(path + "/" + file_name)
        logger.info(f"Image saved as {file_name}")

        save_image_latency.observe(time.time() - start_time)
        return {"message": "Image saved successfully", "filepath": path + "/" + file_name}

    except base64.binascii.Error:
        logger.error("Invalid base64 encoding", exc_info=True)
        error_counter.labels(endpoint="/save-image", error_type="base64_error").inc()
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    except Exception as e:
        logger.error("Error processing image: %s", str(e), exc_info=True)
        error_counter.labels(endpoint="/save-image", error_type="general_error").inc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/invocations/")
def infer_using_mlflow_hosted_model(data: ImgDataForPrediction):
    prediction_requests.inc()
    start_time = time.time()
    logger.info("Received prediction request")
    try:
        base64_encoded_input_image = data.inputs
        payload = {
            "inputs": base64_encoded_input_image
        }
        logger.debug(f"Sending request to MLflow at {mlflow_hosted_model_url}")
        response = requests.post(mlflow_hosted_model_url, json=payload)
        prediction_latency.observe(time.time() - start_time)
        logger.info(f"Prediction response status: {response.status_code}")
        return json.loads(response.text)
    except Exception as e:
        logger.error("Error during model inference: %s", str(e), exc_info=True)
        error_counter.labels(endpoint="/invocations", error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")

@app.get("/metrics")
def metrics():
    logger.debug("Metrics endpoint accessed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
