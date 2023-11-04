import base64
import io
import re
import logging
import time
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model and processor during startup
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class DepthMapRequest(BaseModel):
    image: Union[str, List[str]]  # Union of a single base64-encoded image or a list of base64-encoded images


class DepthMapResponse(BaseModel):
    images: List[str]  # List of base64-encoded depth maps
    info: str

# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"Request start: {request.method} {request.url}")
    response = await call_next(request)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    logger.info(f"Request completed in {elapsed_time} seconds: {request.method} {request.url}")
    return response

# Modify the create_depth_map endpoint
@app.post("/depthmap", response_model=DepthMapResponse)
async def create_depth_map(request: DepthMapRequest):
    
    global device

    depth_maps = []

    if isinstance(request.image, str):
        # If there's only a single image, convert it to a list for uniform processing
        request.image = [request.image]
    
    model.to(device)  # Move the model to the appropriate device

    for base64_image in request.image:
        try:
            # Check if the base64 header is present and remove it if necessary
            match = re.match(r'^data:image/.+;base64,', base64_image)
            if match:
                base64_image = base64_image[len(match.group(0)):]

            # Decode the base64 image
            decoded_image = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(decoded_image))

            logger.info(f"Processing image of size: {image.size}")

            # Process the image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the same device
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

  # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # Convert to image format
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_image = Image.fromarray(formatted)

            # Convert depth image to base64
            buffered = io.BytesIO()
            depth_image.save(buffered, format="PNG")
            base64_depth = base64.b64encode(buffered.getvalue()).decode('utf-8')

            depth_maps.append(base64_depth)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return DepthMapResponse(images=depth_maps, info="Success")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
