import argparse
import json
import os
import torch
from PIL import Image
from torchvision import models, transforms
from utils import load_checkpoint, predict, process_image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained deep learning model")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("--topk", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to a mapping of categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the mapping of categories to real names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the checkpoint and rebuild the model
    model, class_to_idx = load_checkpoint(args.checkpoint)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # Reverse the class_to_idx mapping to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Process the input image
    image = Image.open(args.input)
    image = process_image(image)

    # Perform prediction
    top_probs, top_classes = predict(args.input, model, args.topk)

  

    # Get the class names for the top K classes
    top_class_names = [cat_to_name[str(idx_to_class[class_idx.item()])] for class_idx in top_classes[0]]

    # Print the top classes and probabilities
    for i in range(args.topk):
        class_idx = top_classes[0][i].item()
        class_name = cat_to_name[idx_to_class[class_idx]]
        probability = top_probs[0][i].item()
        print(f"Prediction {i + 1}: {class_name}, Probability: {probability:.4f}")



if __name__ == "__main__":
    main()
    
# "10": "globe thistle",
#python predict.py flowers/valid/10/image_07094.jpg checkpoints/checkpoint.pth --topk 3 --category_names cat_to_name.json --gpu
