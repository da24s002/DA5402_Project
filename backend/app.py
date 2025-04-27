from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import os
from typing import Optional
from PIL import Image
import traceback

import os



app = FastAPI()

class ImageData(BaseModel):
    image: str  # Base64 encoded string
    category: str # string

def update_count_file():
    with open("count.txt", "wb") as f:
        count = int(f.read().strip())
        count += 1
        f.write(count)


@app.post("/save-image/")
async def save_image(data: ImageData):
    try:
        # Decode the base64 string
        image_bytes = base64.b64decode(data.image)
        
        # Convert to numpy array (flattened 28*28 = 784 pixels)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        print(len(image_array))
        
        # Check if the array has the correct size
        if len(image_array) != 784:  # 28*28 pixels
            raise HTTPException(status_code=400, detail="Image must be 28x28 pixels (784 values)")
        
        # Reshape the flattened array to 28x28
        image = image_array.reshape(28, 28)
        parent_path = "uploads/category"
        # Create uploads directory if it doesn't exist
        
        # Save the image as PNG
        path = os.path.join(parent_path, data.category)
        print(path)
        os.makedirs(path, exist_ok=True)
        # To get only files (not directories)
        files_only = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        print(files_only)
        file_name = str(len(files_only)) + ".png"
        # cv2.imwrite(filepath, image)
        img = Image.fromarray(image)
        img.save(path + "/" + file_name)

        update_count_file()
        
        return {"message": "Image saved successfully", "filepath": path + "/" + file_name}
    
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

