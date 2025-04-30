import subprocess
import time
from db_util import create_new_run, update_stage_status, initialize_database, set_stages_not_triggered
import traceback
import logging
from config import file_threshold

logging.basicConfig(
    filename='pipeline_orchestrator_log.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()

break_flag = 0
num_files_generated = 0
stage_num = 1
stage_file_names = [
    "scan_file_count.py", 
    "move_preexisting_new_data_to_old_data.py", 
    "convert_image_to_npy_delete_png.py", 
    "version_data.py", 
    "retrain_model.py",
    "register_latest_model.py",
    "clean_empty_images_folders.py"
]

# file_threshold = 1
logger.info("pipeline triggered")
try:
    initialize_database()
    # Start the pipeline
    run_id = create_new_run()
    logger.info("db initialized")
except Exception as e:
    print(e)
    break_flag = 1
    logger.error("db initialization failed")


# subprocess.run(["python", "scan_file_count.py"])


################### Stage 1 ##################################
if (break_flag == 0):
    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout
        if ("failure" in output.lower()):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            num_files_generated = int(result.stdout)
            if (num_files_generated < file_threshold):
                # Mark stage 1 as complete (success)
                update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
                print(f"stage {stage_num} complete (but not enough files, pipeline will not proceed)")
                logging.info(f"stage {stage_num} complete (but not enough files, pipeline will not proceed)")
                # Mark all subsequent stages as not_triggered
                set_stages_not_triggered(run_id, start_stage=2, end_stage=7)
                break_flag = 1

            else:
                update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
                print(f"stage {stage_num} complete")
                logging.info(f"stage {stage_num} complete")
    except Exception as e:
        traceback(e)
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")
###############################################################
stage_num += 1

#################### Stage 2 ###################################
if (break_flag == 0):
    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout

        if ("failure" in output.lower()):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")
################################################################
stage_num += 1


#################### Stage 3 ###################################
if (break_flag == 0):
    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout

        if ("failure" in output.lower()):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        


################################################################
stage_num += 1


# #################### Stage 3 ###################################
# if (break_flag == 0):
#     result = subprocess.run(["python", "delete_png_files.py"],  capture_output=True, text=True)
#     output = result.stdout

#     if ("failure" in output.lower()):
#         break_flag = 1
# #################################################################

###################### Stage 4 ################################
if (break_flag == 0):

    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout

        if ("failure" in output.lower()):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")


##############################################################
stage_num += 1


###################### Stage 5 ################################
if (break_flag == 0):

    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True, encoding="utf-8")
        end_time = time.time()
        output = result.stdout

        if ("failure" in output):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")


###############################################################
stage_num += 1

###################### Stage 6 ################################
if (break_flag == 0):
    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True, encoding="utf-8")
        end_time = time.time()
        output = result.stdout

        if ("failure" in output):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'failed')
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")

###############################################################
stage_num += 1

# ###################### Stage 7 #################################
# if (break_flag == 0):
#     result = subprocess.run(["python", "deploy_latest_model.py"], capture_output=True, text=True)
#     output = result.stdout

#     if ("failure" in output.lower()):
#         break_flag = 1
#     print("stage 7 complete")
# #################################################################

######################### Stage 7 ################################
if (break_flag == 0):

    try:
        start_time = time.time()
        update_stage_status(run_id, stage_num, 'running')  # Add new status
        result = subprocess.run(["python", stage_file_names[stage_num - 1]], capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout

        if ("failure" in output.lower()):
            update_stage_status(run_id, stage_num, 'failed')
            break_flag = 1
        else:
            update_stage_status(run_id, stage_num, 'complete')
            print(f"stage {stage_num} complete")
            logging.info(f"stage {stage_num} complete")
    except Exception as e:
        update_stage_status(run_id, stage_num, 'complete', start_time, end_time)
        break_flag = 1
        print(f"Stage {stage_num} exception: {e}")
        logging.error(f"Stage {stage_num} exception: {e}")

##############################################################
