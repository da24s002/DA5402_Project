import mlflow
from mlflow.models import infer_signature
import base64
from torchvision import models
import torch
import torchvision.transforms as transforms
import numpy as np
from mlflow import MlflowClient
from config import mlflow_url, experiment_name, classes_file, model_folder_path, best_model_name, deployment_model_name, deployment_model_tag

import logging

logging.basicConfig(
    filename='stage6.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()


mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment(experiment_name)

classes = []


# model_folder_path = "../npy_data/model/"
# best_model_name = "best_model_last_layer_unfreezed.pth"




class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = models.resnet50(pretrained=False)  # Adjust according to your needs

        self.num_classes = 345  # Change to match your number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)

        # 2. Load the state dictionary
        # self.state_dict = torch.load('model_untrained_resnet.pkl')
        self.state_dict = torch.load(model_folder_path + best_model_name)
        self.model.load_state_dict(self.state_dict)

        self.model = self.model.float()
        # 3. Set model to evaluation mode
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize((224, 224)),  # Resize to match ResNet50's expected dimensions
            # transforms.Grayscale(3)         # Convert grayscale to 3-channel by replication
            # Alternatively: transforms.Lambda(lambda x: x.repeat(3, 1, 1)) for tensor inputs
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
        ])

    def predict_val(self, image_tensor):
        # Process your image data to create a tensor
        # ...
        
        # Make sure image_tensor has the right shape and is on the right device
        # image_tensor = image_tensor.double()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            _, predicted_classes = torch.topk(outputs, k=4, dim=1)
            print(predicted_classes.tolist())
            result = predicted.item()  # or whatever processing you need
            result = predicted_classes.tolist()
        
        return result


    def predict(self, model_input):
        try:
            print(model_input, type(model_input), model_input[0][0])
            base64_encoded_str = model_input[0][0]
            image_vector = base64.b64decode(base64_encoded_str)
            numpy_array = np.frombuffer(image_vector, dtype=np.uint8)
            # print(type(image_vector), type(numpy_array))
            # print(numpy_array.shape)
            # Make a writable copy of the array before reshaping
            numpy_array = numpy_array.copy()  # Add this line to create a writable copy 
            # print("flag 1")
            image_vector_reshaped = numpy_array.reshape((28,28,1))
            # print("flag 2")
            image_vector_reshaped_transformed = self.transform(image_vector_reshaped)
            # print("flag 3")
            result = self.predict_val(image_vector_reshaped_transformed)
            print(result)
            # return str(result)
            return [classes[int(v)] for v in result[0]]
            # base64_encoded_str = model_input[0][0]
            # return base64_encoded_str
            # image_vector = base64.b64decode(base64_encoded_str)
            # image_vector_reshaped = image_vector.reshape(28, 28, 1)  # For grayscale with channel dimension
            # # pil_image = Image.fromarray(image_vector_reshaped)
            # image_vector_reshaped_transformed = self.transform(image_vector_reshaped)
            # result = self.predict_val(image_vector_reshaped_transformed)
            
        except Exception as e:
            print(e)
            return "Error in parsing input"

def main():

    global classes
    try:
        f = open(classes_file,"r")
        # And for reading use
        classes = f.readlines()
        f.close()
        classes = [c.replace('\n','').replace(' ','_') for c in classes]

        run_id = ""
        with mlflow.start_run(run_name="doodle-classifier-custom-model") as run:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CustomModel(),
                registered_model_name="doodle-classifier-custom-model",
                signature=infer_signature("str","list")
            )

            run_id = run.info.run_id
            logging.info(f"Current run ID: {run_id}")
            print(f"Current run ID: {run_id}")

        # model_name = "doodle-classifier"
        # model_tag = "model_to_deploy"
        model_name = deployment_model_name
        model_tag = deployment_model_tag

        if run_id != "":
            new_run_id = run_id
            new_model_uri = f"runs:/{new_run_id}/model"

            # Register the new model version
            new_model_details = mlflow.register_model(
                model_uri=new_model_uri,
                name=model_name
            )

            new_version = new_model_details.version
            logging.info(f"Registered new model version: {new_version}")
            print(f"Registered new model version: {new_version}")

            client = MlflowClient()

            # Reassign the "champion" alias to the new version
            client.set_registered_model_alias(
                model_name, 
                model_tag, 
                new_version
            )
    except Exception as e:
        logger.error(f"Exception in stage 6: {e}", exc_info=True)
        print("failure")

main()