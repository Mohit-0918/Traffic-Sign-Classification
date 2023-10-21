# Traffic Sign Classifier using Convolutional Neural Networks (CNN)

This project aims to classify traffic signs using Convolutional Neural Networks (CNN). It utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset for training and testing. The model is built with TensorFlow and Keras, and it achieves high accuracy in classifying various traffic signs.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Model Evaluation](#model-evaluation)
- [Visualizing Model](#visualizing-model)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for training and testing. You can download the dataset from the following link:

[GTSRB Dataset](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)

Please extract the dataset into a folder named 'archive' in the project directory.

## Installation

Make sure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install numpy pandas matplotlib opencv-python tensorflow visualkeras scikit-learn seaborn pillow
```
## Usage

To use the traffic sign classifier, follow these steps:

Clone or download this repository.

Download and extract the GTSRB dataset as mentioned in the Dataset section.
```sh
Make sure your dataset is organized as follows:
├── archive/
    ├── Train.csv
    ├── Test.csv
    ├── Final_Training/Images/
    ├── Final_Test/Images/
```
### Run the provided Python script to build and train the model:
```bash
python3 main.py
```
The above script will load the data from the GTSRB dataset, preprocess it, split it into training and validation sets, build the CNN
The model will train on the training data and provide evaluation metrics. It will also save the trained model as traffic_classifier.h5.

## Model Architecture

The CNN model consists of multiple layers, including convolutional layers, max-pooling layers, dropout layers, and dense layers. The summary of the model architecture can be visualized using the visualkeras library.

## Model Layers
Input Layer:

Type: Convolutional Layer (Conv2D)
Number of Filters: 32
Kernel Size: (5, 5)
Activation Function: ReLU
Input Shape: The shape of the input data is determined by X_train.shape[1:].
Convolutional Layer 2:

Type: Convolutional Layer (Conv2D)
Number of Filters: 32
Kernel Size: (5, 5)
Activation Function: ReLU
Max-Pooling Layer 1:

Type: Max-Pooling Layer (MaxPool2D)
Pool Size: (2, 2)
Dropout Layer 1:

Type: Dropout Layer (Dropout)
Dropout Rate: 0.25
Convolutional Layer 3:

Type: Convolutional Layer (Conv2D)
Number of Filters: 64
Kernel Size: (3, 3)
Activation Function: ReLU
Convolutional Layer 4:

Type: Convolutional Layer (Conv2D)
Number of Filters: 64
Kernel Size: (3, 3)
Activation Function: ReLU
Max-Pooling Layer 2:

Type: Max-Pooling Layer (MaxPool2D)
Pool Size: (2, 2)
Dropout Layer 2:

Type: Dropout Layer (Dropout)
Dropout Rate: 0.25
Flatten Layer:

Type: Flatten Layer (Flatten)
This layer is used to flatten the output from the previous layers into a 1D vector for the fully connected layers.
Dense Layer 1:

Type: Dense Layer (Dense)
Number of Units: 256
Activation Function: ReLU
This layer is fully connected.
Dropout Layer 3:

Type: Dropout Layer (Dropout)
Dropout Rate: 0.5
This dropout layer helps prevent overfitting.
Dense Layer 2 (Output Layer):

Type: Dense Layer (Dense)
Number of Units: 43
Activation Function: Softmax
This is the output layer with 43 units, one for each traffic sign class. 
Softmax activation is used to obtain class probabilities.


These layers make up the Convolutional Neural Network (CNN) model for traffic sign classification. The model is designed to extract features from input images through convolutional layers, reduce dimensionality through max-pooling and prevent overfitting using dropout layers. Finally, it produces class predictions through the softmax output layer.
Training
The model is trained with the following settings:

Number of epochs: 20
Batch size: 64
Loss function: Categorical Cross-Entropy
Optimizer: Adam
You can change these settings in the script according to your requirements.

## Model Evaluation

The model's performance is evaluated on the test dataset. The evaluation includes metrics such as accuracy, loss, classification report, confusion matrix, and a heatmap of the confusion matrix.

## Contributing

Feel free to contribute to this project by creating issues, suggesting enhancements, or making pull requests. Your contributions are highly appreciated.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.