import os
import numpy as np

from sklearn.model_selection import train_test_split
from torchvision import models, transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import model_folder_path, base_npy_file_path, new_npy_file_path, best_model_name, mlflow_url, experiment_name
import mlflow

import logging
import sys

logging.basicConfig(
    filename='stage5.log',  # Replace X with the stage number (e.g., stage1.log)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger()


base_folder_path = base_npy_file_path
new_folder_path = new_npy_file_path
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment(experiment_name)
# model_folder_path = "../npy_data/model/"

classes = []
selection_size = 10


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Custom dataset that handles reshaping and transformations
class FlattenedImageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features  # Shape: (n_samples, 784)
        self.labels = labels
        self.transform = transform
        self.img_size = int(np.sqrt(features.shape[1]))  # Assuming square images (28×28)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Reshape flattened image to 2D (height, width)
        img = self.features[idx].reshape(self.img_size, self.img_size)
        
        # Convert to PIL Image for transformations
        img = transforms.ToPILImage()(img.astype(np.uint8))
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)
        
        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label


def create_data_set():
    X, y = np.array([]), np.array([])

    base_npy_files = [f for f in os.listdir(base_folder_path) if f.endswith('.npy')]
    new_npy_files = [f for f in os.listdir(new_folder_path) if f.endswith('.npy')]

    choice_of_class_indices = np.random.choice(345,100)



    for index in choice_of_class_indices:
        temp_arr = np.load(base_folder_path + classes[index] + ".npy")

        no_of_rows = len(temp_arr)

        indices = np.random.choice(no_of_rows, selection_size)
        selection_matrix = temp_arr[indices ]

        if (len(X) == 0):
            X = np.copy(selection_matrix)
            y = np.array([index] * selection_size)

        else:
            X = np.vstack((X, selection_matrix))
            y = np.concatenate((y, np.array([index] * selection_size)))

    for new_npy_file in new_npy_files:
        class_name = new_npy_file[:len(new_npy_file) - len(".npy")]
        index = classes.index(class_name)
        temp_arr = np.load(new_folder_path + new_npy_file)

        X = np.vstack((X, temp_arr))
        y = np.concatenate((y, np.array([index] * len(temp_arr))))

    return X, y

def main():

    global classes
    try:
        f = open("categories.txt","r")
        # And for reading use
        classes = f.readlines()
        f.close()

        classes = [c.replace('\n','').replace(' ','_') for c in classes]

        
        X, y = create_data_set()
        logger.info("data set created by sampling and taking new data points")
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)


        # best_model_name = "best_model_last_layer_unfreezed.pth"
        model_complete_path = model_folder_path + best_model_name

        # 1. Load your saved model
        model = models.resnet50()
        logger.info("resnet 50 model loaded")


        num_classes = 345
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Load the saved weights
        checkpoint = torch.load(model_complete_path)
        model.load_state_dict(checkpoint)

        # 2. Freeze all layers except the final layer
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the final layer
        for param in model.fc.parameters():
            param.requires_grad = True

        # 3. Set up optimizer for only the final layer
        optimizer = optim.Adam(model.fc.parameters())#, lr=0.001, momentum=0.9)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((28,28)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize((224, 224)),  # Resize to match ResNet50's expected dimensions
            # transforms.Grayscale(3)         # Convert grayscale to 3-channel by replication
            # Alternatively: transforms.Lambda(lambda x: x.repeat(3, 1, 1)) for tensor inputs
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
        ])

        # Create dataset
        train_dataset = FlattenedImageDataset(X_train, y_train, transform=transform)

        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation dataset and loader
        val_dataset = FlattenedImageDataset(X_val, y_val, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



        # 5. Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        best_val_accuracy = 0

        num_epochs = 10
        logger.info("training is starting")
        with mlflow.start_run("retrain-doodle-classifier"):
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in tqdm(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                # Calculate training metrics
                train_loss = running_loss / len(train_loader)
                train_accuracy = 100 * correct / total
                
                # Validation phase
                val_loss, val_accuracy = validate(model, val_loader, criterion, device)

                if (val_accuracy > best_val_accuracy):
                    torch.save(model.state_dict(), model_complete_path)
                    best_val_accuracy = val_accuracy
                
                # Print statistics
                # Log metrics for this epoch
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        logger.info("training ends")
    except Exception as e:
        print("failure")
        logger.error(f"Exception in stage 5: {e}", exc_info=True)
        

# Save the fine-tuned model
# torch.save(model.state_dict(), 'fine_tuned_model.pth')

main()