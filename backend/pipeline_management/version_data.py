import subprocess
from config import dvc_folder_tracked

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# result = subprocess.run(["dvc", "commit", "npy_data"], cwd=parent_dir, capture_output=True, text=True)
# result = subprocess.run(["git", "add", "folder_name.dvc"], capture_output=True, text=True)
# result = subprocess.run(["git", "commit", "-m", "'Update tracked data in folder_name'"], capture_output=True, text=True)



import subprocess

import logging
import sys

logging.basicConfig(
    filename='stage4.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()

def run_command(command, description):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        # print(f"{description} completed")
        logger.info(f"{description} completed")
        if result.stdout.strip():
            # print(f"Output: {result.stdout.strip()}")
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        # print(f"{description} failed with return code {e.returncode}")
        # print(f"Error: {e.stderr.strip()}")
        logger.error(f"{description} failed with return code {e.returncode}")
        logger.error(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        # print(f"Command not found: {command[0]}")
        # print(f"Make sure {command[0]} is installed and in your PATH")
        logger.error(f"Command not found: {command[0]}")
        logger.error(f"Make sure {command[0]} is installed and in your PATH")
        return False
    except subprocess.TimeoutExpired:
        # print(f"{description} timed out")
        logger.error(f"{description} timed out")
        return False
    except Exception as e:
        # print(f"Unexpected error during {description}: {str(e)}")
        logger.error(f"Unexpected error during {description}: {str(e)}")
        return False

def update_dvc_tracking(folder_name):
    # Make sure we're in a git repository
    # if not os.path.exists(".git"):
    #     print("Not in a git repository. Please run this script from the root of your git repository.")
    #     return False
    
    # Step 1: Commit changes to DVC
    if not run_command(
        ["dvc", "add", folder_name],
        "DVC add"
    ):
        return False
    else:
        # print("dvc added")
        logger.info("dvc added")
    
    # Step 2: Add the .dvc file to git
    if not run_command(
        ["git", "add", f"{folder_name}.dvc"],
        "Git add"
    ):
        return False
    else:
        # print("git added")
        logger.info("git added")
    
    # Step 3: Commit the changes to git
    return run_command(
        ["git", "commit", "-m", f"Update tracked data in {folder_name}"],
        "Git commit"
    )

if __name__ == "__main__":
    # Replace this with your actual folder name
    folder_to_update = dvc_folder_tracked
    try:
        if update_dvc_tracking(folder_to_update):
            print("All operations completed successfully")
            logger.info("All operations completed successfully")
        else:
            print("Failed to update DVC tracking, failure")
            logger.error("Failed to update DVC tracking, failure")
    except Exception as e:
        logger.error(f"Exception in stage 4: {e}", exc_info=True)
        print("failure")
        