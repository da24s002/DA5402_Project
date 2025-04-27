import subprocess

break_flag = 0
num_files_generated = 0

# subprocess.run(["python", "scan_file_count.py"])


################### Stage 1 ##################################
result = subprocess.run(["python", "scan_file_count.py"], capture_output=True, text=True)
if ("failure" in result.stdout):
    break_flag = 1
else:
    num_files_generated = int(result.stdout)
    if (num_files_generated < 10):
        break_flag = 1
###############################################################

#################### Stage 2 ###################################
if (break_flag == 0):
    result = subprocess.run(["python", "convert_image_to_npy_delete_png.py"], capture_output=True, text=True)
    output = result.stdout

    if ("failure" in output.lower()):
        break_flag = 1
################################################################

# #################### Stage 3 ###################################
# if (break_flag == 0):
#     result = subprocess.run(["python", "delete_png_files.py"],  capture_output=True, text=True)
#     output = result.stdout

#     if ("failure" in output.lower()):
#         break_flag = 1
# #################################################################

###################### Stage 3 ################################
if (break_flag == 0):
    result = subprocess.run(["python", "version_data.py"], capture_output=True, text=True)
    output = result.stdout

    if ("failure" in output.lower()):
        break_flag = 1
###############################################################

# ###################### Stage 4 ################################
# if (break_flag == 0):
#     result = subprocess.run(["python", "retrain_model.py"], capture_output=True, text=True)
#     output = result.stdout

#     if ("failure" in output.lower()):
#         break_flag = 1
# ###############################################################

# ###################### Stage 5 #################################
# if (break_flag == 0):
#     result = subprocess.run(["python", "deploy_latest_model.py"], capture_output=True, text=True)
#     output = result.stdout

#     if ("failure" in output.lower()):
#         break_flag = 1
# #################################################################