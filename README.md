# Image Classifier Project

  This project is part of the Udacity AI programming with Python Nanodegree program. It involves training an image classifier to recognize different species of flowers. The project includes training a deep learning model, optimizing it, and providing a command-line interface for making predictions with the trained model.
  
  ## Table of Contents
  - [Project Overview](#project-overview)
  - [Requirements](#requirements)
  - [Getting Started](#getting-started)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
  - [Optimizations](#optimizations)
  - [Command-Line Interface](#command-line-interface)
  - [Files Submitted](#files-submitted)
  - [Development Notebook](#development-notebook)
  - [Command Line Application](#command-line-application)
  - [License](#license)

## Project Overview

  The goal of this project is to build an image classifier capable of recognizing different species of flowers. The project consists of two main components: training the deep learning model and providing a command-line interface for making predictions with the trained model.

## Requirements

  - Python 3.x
  - PyTorch
  - torchvision
  - PIL (Pillow)
  - argparse
  - json

### You can install the required packages using `pip`:
  pip install torch torchvision pillow argparse

## Getting Started
  1. Clone this repository to your local machine:
    git clone https://github.com/your-username/image-classifier.git
    cd image-classifier
    
  2. Download and prepare the flower dataset. You can use the provided download_data.sh script to download and unzip the dataset.
    sh download_data.sh

## Training the Model
  - To train the model, you can use the train.py script. You can choose between two architectures: VGG16 and DenseNet121.
  - The trained model will be saved as a checkpoint in the checkpoints directory.

  ### to run the script:
    python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
    
  #### Notes on my trained model:
  In this project, I've implemented several techniques to efficiently train the image classifier model while optimizing the use of resources:
    
    1. Loading Pre-trained VGG16 Model: I started by loading a pre-trained VGG16 model from torchvision, leveraging the pre-trained weights on the ImageNet dataset. This step provides me with a powerful feature extractor.
    
    2. Freezing Pre-trained Parameters: To prevent retraining the entire model, I froze the parameters of the pre-trained VGG16 model. This means that during backpropagation, gradients are not computed for these layers.
    
    3. Custom Classifier Modification: I then modified the classifier part of the VGG16 model to suit my specific task. A new feedforward network was defined as the classifier, which consists of fully connected layers, ReLU activation, and dropout to prevent overfitting.
    
    4. Loss Function and Optimizer: I defined a loss function (Cross-Entropy Loss) and an optimizer (Adam) to train the classifier. The optimizer is applied only to the parameters of the classifier, as the feature extractor's parameters are frozen.
    
    5. Learning Rate Scheduler: To improve training stability and convergence, I implemented a learning rate scheduler. This scheduler adjusts the learning rate during training, potentially speeding up convergence.
    
    6. Data Augmentation and Normalization: For the training data, I applied data augmentation techniques such as random resizing, cropping, and horizontal flipping. These augmentations introduce diversity into the dataset, making the model more robust. Additionally, I performed data normalization to ensure consistent input scales, improving training efficiency.
    
    7. Mini-Batch Gradient Accumulation: To balance GPU memory usage and efficient training, I introduced mini-batch gradient accumulation. This technique allows me to accumulate gradients over multiple mini-batches before performing an optimization step. It simplifies training and improves memory management.
  ![Screenshot 2023-10-05 at 3 44 14 PM](https://github.com/yannellym/ImageClassifier/assets/91508647/9e7f7d1f-2f83-4468-86c4-9e512ac4ab54)

### Model performance when tested: 

    
## Optimizations
 Several optimizations have been made in this project, including:

  💡 Cropping and Normalization: Cropping and normalizing the datasets according to the ImageNet standard is essential for various reasons:
  
  Consistent Input Size: In many machine learning models, including transformers, it is important to have a consistent input size. By cropping the data, we ensure that all input samples have the same dimensions. This is particularly useful when working with sequences of varying lengths, such as sentences or paragraphs. Cropping the data to a fixed length allows us to efficiently process the data in batches and train the model more effectively.
  
  Removal of Unnecessary Information: Cropping the data involves selecting a specific portion or window of the input. This helps in removing unnecessary information from the input sequence that may not be relevant to the task at hand. For example, when translating a sentence, we can crop the input to focus only on the source sentence and exclude any additional context or noise.
  
  Normalization for Improved Training: Normalization is the process of scaling the input data to a standard range. It helps in improving the convergence and stability of the training process. By normalizing the data, we ensure that the input features have similar scales, preventing certain features from dominating the learning process. This is particularly important when working with numerical or continuous features.
  
  Mitigating the Impact of Outliers: Normalization can also help in reducing the impact of outliers or extreme values in the data. Outliers can skew the learning process and affect the model's ability to generalize. By normalizing the data, we bring the values within a specific range, making the model more robust to outliers and improving its performance.
  
  Gradient Descent Optimization: Normalizing the data can aid in the optimization process, especially when using gradient-based optimization algorithms. It helps in achieving faster convergence by ensuring that the gradients are within a similar range across different features. This can lead to more stable and efficient training.

## Making Predictions
To make predictions using the trained model, you can use the predict.py script. You need to specify the path to an image and the checkpoint file. The following rubric points are addressed in this section:

### Command Line Application
Training a network
Training validation log
Model architecture
Model hyperparameters
Training with GPU
Predicting classes
Top K classes
Displaying class names
Predicting with GPU

python predict.py path/to/image.jpg checkpoints/checkpoint.pth --topk 3 --category_names cat_to_name.json --gpu

## Files Submitted
The following files are included:

Part 1 - Development Notebook
Development Notebook
truncated 'flowers' folder.

Part 2 - Command Line Application
train.py
predict.py
