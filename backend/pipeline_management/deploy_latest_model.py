import subprocess
import sys
from mlflow.tracking import MlflowClient

def get_latest_model_version(model_name):
    try:
        client = MlflowClient()
        model_versions = client.get_latest_versions(model_name)
        latest_version = max([int(version.version) for version in model_versions])
        # logging.info("fetched latest model no.")
        return latest_version
    except Exception as e:
        # logging.error("error in getting latest model version")
        # logging.error(e)
        pass

def kill_port_windows(port):
    try:
        # Find the process ID using netstat
        result = subprocess.run(
            ["netstat", "-ano", "|", "findstr", f":{port}"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        # Parse the output to get the PID
        lines = result.stdout.splitlines()
        pid_to_kill = None
        
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                pid_to_kill = line.split()[-1]
                break
                
        if pid_to_kill:
            # Kill the process using taskkill
            kill_result = subprocess.run(
                ["taskkill", "/PID", pid_to_kill, "/F"],
                capture_output=True,
                text=True
            )
            
            if kill_result.returncode == 0:
                print(f"Process on port {port} with PID {pid_to_kill} has been successfully killed.")
                return True
            else:
                print(f"Failed to kill the process on port {port}. Error: {kill_result.stderr}")
                return False
        else:
            print(f"No process found running on port {port}.")
            return True
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def switch_mlflow_model(new_version, latest_model_name, port=5002):
    # Kill the existing MLflow service
    if kill_port_windows(port):
        # Start the new MLflow service
        try:
            cmd = [
                sys.executable, "-m", "mlflow", "models", "serve",
                "-m", f"models:/{latest_model_name}/{new_version}",
                "--port", str(port),
                "--host", "0.0.0.0",
                "--env-manager=conda"
            ]
            # Use DETACHED_PROCESS flag on Windows to detach the process
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            print(f"Started MLflow serve for model version {new_version} on port {port}")
            return True  # Return True instead of the process
        except Exception as e:
            print(f"Failed to start new model service: {e}")
            return False
    else:
        print("Could not kill the existing service. Model switch aborted.")
        return False

if __name__ == "__main__":
    model_name = "doodle-classifier-custom-model"
    # Example usage: Switch from model version 19 to 20
    latest_model_version = get_latest_model_version(model_name)
    success = switch_mlflow_model(latest_model_version, model_name, 5002)
    if success:
        print("success")  # Print success for the pipeline to detect
    else:
        print("failure")  # Print failure for the pipeline to detect
