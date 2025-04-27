import os
import numpy as np
from PIL import Image

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
    
    return category_data

def main():
    # Get all directories in the current folder
    root_dir = './uploads/category'  # Current directory, change if needed
    categories = [d for d in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, d))]
    
    # print(f"Found {len(categories)} categories: {categories}")
    
    # Process each category
    for category in categories:
        category_path = os.path.join(root_dir, category)
        # print(f"Processing {category}...")
        
        # Get the data matrix for this category
        category_data = process_category_folder(category_path)
        
        # Save as .npy file
        output_file = f"{category}.npy"
        np.save(output_file, category_data)
        
        # print(f"Saved {len(category_data)} images to {output_file}, shape: {category_data.shape}")
    print("success")

if __name__ == "__main__":
    main()
