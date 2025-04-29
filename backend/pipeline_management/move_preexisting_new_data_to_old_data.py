import numpy as np
import os
from config import old_npy_file_path, new_npy_file_path

import logging
import sys

logging.basicConfig(
    filename='stage2.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()

# old_npy_file_path = "../npy_data/uploaded_data/old_data/"
# new_npy_file_path = "../npy_data/uploaded_data/new_data/"

def main():
    try:
        old_npy_files = [f for f in os.listdir(old_npy_file_path) if f.endswith('.npy')]
        new_npy_files = [f for f in os.listdir(new_npy_file_path) if f.endswith('.npy')]
        for file in new_npy_files:
            if file not in old_npy_files:
                data = np.load(new_npy_file_path + file)
                np.save(old_npy_file_path + file, data)
                logger.info(f"Moved new file {file} to old data")
            else:
                new_data = np.load(new_npy_file_path + file)
                old_data = np.load(old_npy_file_path + file)
                total_data = np.vstack((old_data, new_data))
                np.save(old_npy_file_path + file, total_data)
                logger.info(f"Merged new data into {file}")
            os.remove(new_npy_file_path + file)
        logger.info("Stage 2 success")
        print("success")
    except Exception as e:
        logger.error(f"Exception in stage 2: {e}", exc_info=True)
        print("failure")


main()