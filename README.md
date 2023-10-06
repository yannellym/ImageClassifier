# Image Classifier Project

  This project is part of the Udacity AI programming with Python Nanodegree program. It involves training an image classifier to recognize different species of flowers. The project includes training a deep learning model, optimizing it, and providing a command-line interface for making predictions with the trained model.
  
  ## Table of Contents
  - [Project Overview](#project-overview)
  - [Requirements](#requirements)
  - [Getting Started](#getting-started)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
  - [Optimizations](#optimizations)
  - [Command-Line App](#command-line-app)
  - [Prediction on unknown data](#prediction-on-unknown-data)
  - [Conclusion](#conclusion)

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

### Making Predictions: 
![Screenshot 2023-10-05 at 3 44 52 PM](https://github.com/yannellym/ImageClassifier/assets/91508647/88e382a7-e537-450b-a246-bc8a493f4095)

    
## Optimizations
 Several optimizations have been made in this project, including:

  1. Cropping and Normalization: Cropping and normalizing the datasets according to the ImageNet standard is essential for various reasons:
  
  2. Consistent Input Size: In many machine learning models, including transformers, it is important to have a consistent input size. By cropping the data, we ensure that all input samples have the same dimensions. This is particularly useful when working with sequences of varying lengths, such as sentences or paragraphs. Cropping the data to a fixed length allows us to efficiently process the data in batches and train the model more effectively.
  
  3. Removal of Unnecessary Information: Cropping the data involves selecting a specific portion or window of the input. This helps in removing unnecessary information from the input sequence that may not be relevant to the task at hand. For example, when translating a sentence, we can crop the input to focus only on the source sentence and exclude any additional context or noise.
  
  4. Normalization for Improved Training: Normalization is the process of scaling the input data to a standard range. It helps in improving the convergence and stability of the training process. By normalizing the data, we ensure that the input features have similar scales, preventing certain features from dominating the learning process. This is particularly important when working with numerical or continuous features.
  
  5. Mitigating the Impact of Outliers: Normalization can also help in reducing the impact of outliers or extreme values in the data. Outliers can skew the learning process and affect the model's ability to generalize. By normalizing the data, we bring the values within a specific range, making the model more robust to outliers and improving its performance.
  
  6. Gradient Descent Optimization: Normalizing the data can aid in the optimization process, especially when using gradient-based optimization algorithms. It helps in achieving faster convergence by ensuring that the gradients are within a similar range across different features. This can lead to more stable and efficient training.

- Example:
![Screenshot 2023-10-06 at 1 04 26 PM](https://github.com/yannellym/ImageClassifier/assets/91508647/cc844190-5f30-4e05-b2ed-c8087902be79)

## Command Line App

### Building the Command Line Application

In the second part of this project, I've created a command-line application consisting of two Python scripts: `train.py` and `predict.py`. These scripts empower users to interact with the trained deep neural network for image classification.

#### Train.py

To train a new neural network on a dataset using `train.py`, you can use basic usage or customize its behavior with various options:

- **Basic Usage**: `python train.py data_directory`
  - This command prints training loss, validation loss, and validation accuracy as the network trains.

- **Options**:
  - Set the directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
  - Choose the architecture (e.g., "vgg13"): `python train.py data_dir --arch "vgg13"`
  - Adjust hyperparameters (e.g., learning rate, hidden units, epochs): `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
  - Utilize GPU for training: `python train.py data_dir --gpu`

#### Predict.py

With `predict.py`, you can predict the flower name and its associated probability for a given input image. Similar to `train.py`, it offers basic usage and various options:

- **Basic Usage**: `python predict.py /path/to/image checkpoint`
  - This command returns the top K most likely classes.

- **Options**:
  - Specify the number of top classes to return: `python predict.py input checkpoint --top_k 3`
  - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
  - Enable GPU for inference: `python predict.py input checkpoint --gpu`

Both scripts make use of the argparse module for efficient command-line argument parsing, ensuring a user-friendly experience for image classification tasks.
- Example of train.py:
<img width="1057" alt="Screenshot 2023-10-06 at 11 38 37 AM" src="https://github.com/yannellym/ImageClassifier/assets/91508647/9f4c39f2-0a00-4cd8-951e-b95fe5b3be4a">

- Example of predict.py: 
<img width="907" alt="Screenshot 2023-10-06 at 12 07 58 PM" src="https://github.com/yannellym/ImageClassifier/assets/91508647/0154dd13-803f-4593-95e9-d9bcd1495117">

##  Prediction on unknown data:
  The model achieved impressive accuracy by correctly identifying an image of a petunia during testing. This showcases its proficiency in recognizing distinct flower features and highlights its potential for various image recognition tasks related to different species.
<img width="592" alt="Screenshot 2023-10-06 at 1 12 22 PM" src="https://github.com/yannellym/ImageClassifier/assets/91508647/deba00f6-8a1f-4400-a6e7-e15c332cfb44">

## The following files are included:

Part 1 - Development Notebook
Development Notebook
truncated 'flowers' folder.

Part 2 - Command Line Application
train.py
predict.py

## Conclusion

I thoroughly enjoyed working on this project, which allowed me to delve into the fascinating world of deep learning and image classification. The journey from training a model to optimizing its performance and building a command-line interface was both challenging and rewarding. I've gained valuable insights into working with neural networks and learned numerous techniques for improving model efficiency and accuracy. This project has not only expanded my knowledge but also ignited my passion for AI and computer vision. I look forward to applying these newfound skills to more exciting projects in the future.

