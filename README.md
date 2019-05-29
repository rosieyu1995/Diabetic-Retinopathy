# Diabetic Retinopathy Detection and Prevention by deep learning

Diabetic retinopathy is a progressive disease that is classified into one of 5 stages by an ophthalmologist, based on severity. The disease and its risk factors, as well as its symptoms, are well understood. It is diagnosed by an ophthalmologist, who examines features discovered via a visual examination and/or fundus photography (photography of the back of the eye), or other forms of optical imaging.

## Data
The data is from Kaggle(https://www.kaggle.com/c/classroom-diabetic-retinopathy-detection-competition/data). The data contains training and validation sets. This data contains images that are taken of different people, some are flipped and some are not, and the brightness of each image is different. Therefore, we do some preprocessing steps and data argumentation to get all images to a useable format for training a model.

Class | Training Data Counts | Validation Data Counts
---- | ---- | ---- |
Class 0| 23229 | 2581  |
Class 1| 2199  | 244   |
Class 2| 4763  | 529   |
Class 3| 786   | 87    |
Class 4| 639   | 70    |

## Prerequisites

- Python 2.7 or 3.6 with tensorflow as backend
- Keras
- Numpy

## Data Preprocessing and Data Argumentation

By modifying and standardizing the images we had obtained, we used ImageDataGenerator. From the ImageDataGenerator’s `rescale` argument, we passed (1. / 255) to normalize values from 0 to 1 in a grid of 256. We also set `shear_range` to equal 0.2, which is the shear intensity – the amount of image that we will shear off. We then set `zoom_range` to be 0.2, which is the range for random zoom – an image may be zoomed in or out on from a range of 0.80 to 1.2, at random. 
We also set `horizontal_flip = True`, which will randomly flip some of our images horizontally – this is useful in this case in particular because fundus photographs of the left and right eyes are mirrored, so we dealt with this by flipping a portion of the images at random.
We then defined train_generator to equal `train_datagen.flow_from_directory( )`, which will generate labels for images from a target directory. Our target_size is (224, 224), the dimensions of the images we would like to generate from our directory of images. We set `batch_size` to equal the default of 32, and we set class_mode to equal “categorical”, which determines that the type of label array returned is a 2D array. We also defined `validation_generator` in a similar way, but changed the directory to the directory for our validation images.

## Neural Network Architectures

We defined our model as Sequential( ), with the first layer being a `Conv2D layer` of 32 neurons and a (3x3) kernel with padding = 'same', and our input shape correspondingly being [224, 224, 3]. For this and all subsequent Conv2D layers, `kernel_regularizer` was set to equal regularizers.12((0.0001) – this was done to apply a penalty on our layer parameters during its optimization, which would be included in our loss function. Our activation function for this layer was `relu`. The next layer was a `MaxPooling2D layer` of pool size (2, 2), followed by another `Conv2D layer` of (32, (3x3)) and another `MaxPooling2D layer` with the same pool size as before.  Our final `Conv2D layer` was of (64, (3x3)) and was followed by one more `MaxPooling2D layer`. We then `flattened` the output of this MaxPoolingLayer and fed it into a `Dense layer` of 256 neurons with a relu activation function. We used a `Dropout layer` with 0.5 dropout, and then one more Dense layer of 5 neurons as our final layer. The activation function for this layer was the `softmax function`, since we were looking to output 5 continuous probabilities from 0 to 1 for each of 5 classes. Our final model had 12.875 million trainable parameters.

## Results

We used an early stopping function that monitored loss in our validation set with a patience of 2 (the number of epochs with no improvement that would lead to stopping training). 
After compiled and fitting the model, our testing accuracy on our withheld images using this model was **73.73%**.

## Project Participants

- Ho Huang
- Spencer Glass


