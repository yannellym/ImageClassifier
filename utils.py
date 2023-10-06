import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import numpy as np
from PIL import Image
import torch.nn.functional as F 

def load_data(data_dir):
    """
    Load and preprocess the data.

    Args:
        data_dir (str): Directory containing the data.

    Returns:
        dataloaders (dict): Data loaders for training, validation, and testing.
        dataset_sizes (dict): Sizes of the training, validation, and testing datasets.
        class_names (list): List of class names.
        class_to_idx (dict): Mapping of class names to class indices.
    """
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets with transformations
    image_datasets = {
        x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # Create dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
        for x in ['train', 'valid', 'test']
    }

    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    # Define the class_to_idx mapping
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, dataset_sizes, class_names, class_to_idx


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, use_gpu=False, save_dir=None):
    """
    Train a model on the given dataset.

    Args:
        model (nn.Module): The neural network model to train.
        dataloaders (dict): Data loaders for training and validation.
        criterion (nn.Module): Loss criterion.
        optimizer (optim.Optimizer): Optimizer for training.
        device (str): Device to use for training ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.

    Returns:
        model (nn.Module): Trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{epoch}. {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_epoch = epoch

        # Check if we have reached the desired number of epochs
        if epoch == num_epochs - 1:
            print("Model is now done training")
            break

    return model, best_epoch

def process_image(image):
    # Open and resize the image while maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = 256
        new_height = int(256 * height / width)
    else:
        new_width = int(256 * width / height)
        new_height = 256

    image = image.resize((new_width, new_height))

    # Crop the center of the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert color channels to values in the range [0, 1]
    image = np.array(image) / 255.0

    # Normalize the image with mean and standard deviation of ImageNet data
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # Reorder color channel dimensions
    image = image.transpose((2, 0, 1))

    # Convert the NumPy array back to a PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32)

    return image

def save_checkpoint(model, optimizer, class_names, class_to_idx, checkpoint_path, architecture):
    """
    Save the trained model checkpoint.

    Args:
        model (nn.Module): Trained model.
        optimizer (optim.Optimizer): Optimizer used for training.
        class_names (list): List of class names.
        class_to_idx (dict): Mapping of class names to class indices.
        checkpoint_path (str): Path to save the checkpoint.
        architecture (str): The architecture name (e.g., 'vgg16' or 'densenet121').
    """
    # Create a dictionary to store checkpoint information
    checkpoint = {
        'architecture': architecture,  # Save the architecture name
        'model_state_dict': model.state_dict(),  # Save the model state dictionary
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'class_to_idx': class_to_idx,  # Include the class_to_idx mapping
        'classifier': model.classifier  # Include the classifier
    }
    
    # Save the checkpoint to the specified checkpoint_path
    torch.save(checkpoint, checkpoint_path)




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')  # Load the checkpoint
    print("Loaded architecture:", checkpoint['architecture'])  # Print the architecture name
    
    # Create the model with the correct architecture
    model = models.__dict__[checkpoint['architecture']](pretrained=True)
    
    # Recreate the classifier based on the checkpoint's classifier
    model.classifier = checkpoint['classifier']

    # Load the model's state_dict and optimizer's state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class-to-index mapping
    class_to_idx = checkpoint['class_to_idx']

    return model, class_to_idx



def predict(image_path, model, topk=1, device="cpu"):
    image = Image.open(image_path)
    image = process_image(image).unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        probabilities, indices = torch.topk(F.softmax(output, dim=1), topk)  # Use F.softmax
    
    return probabilities, indices