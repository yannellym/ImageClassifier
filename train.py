import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
from utils import load_data, train_model, save_checkpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a deep learning model for image classification")
    parser.add_argument("data_directory", help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Architecture (vgg16 or densenet121)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load and preprocess the data
    dataloaders, dataset_sizes, class_names, class_to_idx = load_data(args.data_directory)

    # Load the pre-trained model
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_inputs = 25088
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True)
        num_inputs = 1024
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'densenet121'.")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build custom classifier
    classifier = nn.Sequential(
        nn.Linear(num_inputs, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, len(class_to_idx))
    )
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print(f"Model is now training. Total number of epochs: {args.epochs}")

    # Train the model
    model, best_epoch = train_model(
        model,
        criterion,
        optimizer,
        dataloaders,
        dataset_sizes,
        args.epochs,
        use_gpu=args.gpu,
        save_dir=args.save_dir
    )

    # Save the checkpoint using the save_checkpoint function
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    save_checkpoint(model, optimizer, class_names, class_to_idx, checkpoint_path, args.arch)

if __name__ == "__main__":
    main()

    #python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
