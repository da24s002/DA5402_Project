import os
from config import img_data_location
# img_data_location = "../uploaded_img_data/category"
import logging
import sys

logging.basicConfig(
    filename='stage7.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()


def delete_empty_subfolders(parent_folder):
    try:
        # List all items in the parent folder
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Check if the directory is empty
                if not os.listdir(item_path):
                    # Delete the empty directory
                    os.rmdir(item_path)
                    # print(f"Deleted empty folder: {item_path}")
                    logger.info(f"Deleted empty folder: {item_path}")
        logger.info("Stage 7 success")
    except Exception as e:
        logger.error(f"Exception in stage 7: {e}", exc_info=True)
        print("failure")

# Example usage
parent_folder = img_data_location  # Change this to your actual folder path
delete_empty_subfolders(parent_folder)