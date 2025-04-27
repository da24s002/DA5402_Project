import os
from common_method import trigger_script
import subprocess

img_data_location = "./uploaded_img_data/category"

def get_count_img_in_category(folder_path):
    # Get all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # print(image_files)
    return len(image_files)


def main():
    total_count = 0
    root_dir = img_data_location  # Current directory, change if needed
    categories = [d for d in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, d))]
    
    # print(f"Found {len(categories)} categories: {categories}")
    
    # Process each category
    for category in categories:
        category_path = os.path.join(root_dir, category)
        
        # Get the data matrix for this category
        no_of_files = get_count_img_in_category(category_path)
    
        total_count += no_of_files
    print(total_count)

    

main()