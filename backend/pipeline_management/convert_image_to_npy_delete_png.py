import os
import numpy as np
from PIL import Image
from config import img_data_location, new_npy_file_path

import logging
import sys

logging.basicConfig(
    filename='stage3.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()

# img_data_location = "../uploaded_img_data/category"
# npy_new_data_location = "../npy_data/uploaded_data/new_data/"
npy_new_data_location = new_npy_file_path

def process_category_folder(folder_path):
    # Get all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Create an empty array to store flattened images
    # Each image will be a row in this array
    category_data = np.zeros((len(image_files), 784), dtype=np.uint8)
    
    # Process each image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        
        # Open image and convert to grayscale if it's not already
        img = Image.open(img_path).convert('L')
        
        # Resize to 28x28 if needed
        if img.size != (28, 28):
            img = img.resize((28, 28))
        
        # Convert to numpy array and flatten
        img_array = np.array(img)
        flattened_img = img_array.reshape(784)
        
        # Add to our data matrix
        category_data[i] = flattened_img
    
    return category_data, image_files

def main():
    try: 
        # Get all directories in the current folder
        root_dir_img = img_data_location  # Current directory, change if needed
        root_dir_npy = npy_new_data_location
        categories = [d for d in os.listdir(root_dir_img) 
                    if os.path.isdir(os.path.join(root_dir_img, d))]
        
        # print(f"Found {len(categories)} categories: {categories}")
        logger.info(f"Processing categories: {categories}")
        # Process each category
        for category in categories:
            category_path = os.path.join(root_dir_img, category)
            # print(f"Processing {category}...")
            
            # Get the data matrix for this category
            category_data, image_files = process_category_folder(category_path)
            
            # Save as .npy file
            output_file = f"{category}.npy"
            output_file_path = root_dir_npy + output_file
            np.save(output_file_path, category_data)

            for image_file in image_files:
                os.remove(os.path.join(category_path, image_file))
            logger.info(f"Saved and cleaned up category {category}")
        logger.info("Stage 3 success")
            
            # print(f"Saved {len(category_data)} images to {output_file}, shape: {category_data.shape}")
        print("success")
    except Exception as e:
        logger.error(f"Exception in stage 3: {e}", exc_info=True)
        print("failure")

if __name__ == "__main__":
    main()
