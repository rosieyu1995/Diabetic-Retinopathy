# Introduction

Diabetic retinopathy is a progressive disease that is classified into one of 5 stages by an ophthalmologist, based on severity. The disease and its risk factors, as well as its symptoms, are well understood. It is diagnosed by an ophthalmologist, who examines features discovered via a visual examination and/or fundus photography (photography of the back of the eye), or other forms of optical imaging.

## Data
The data is from Kaggle(https://www.kaggle.com/c/classroom-diabetic-retinopathy-detection-competition/data). However, this data contains images that are taken of different people, some are flipped and some are not. Therefore we do some preprocessing steps to get all images to a useable format for training a model.

The training data within different classes:
| Class        | Count |
| ------------ | ----- |
| Class 0      | 23229 |
| Class 1      | 2199  |
| Class 2      | 47623 |
| Class 3      | 786   |
| Class 4      | 639   |

The testing data within different classes:
| Class        | Count |
| ------------ | ----- |
| Class 0      | 2581  |
| Class 1      | 244   |
| Class 2      | 529   |
| Class 3      | 87    |
| Class 4      | 70    |

## Getting Started

The project uses Convolutional Neural Networks(CNN) with 5 softmax output layers to classify retina photographs into one of the five levels on the clinical scale of diabetic retinopathy and ImageDataGenerator to standardize the images, adjust the amount of image that we will shear off, randomly zoom in or out of a image and flip some of our images horizontally.

### Prerequisites

Python 2.7 or 3.6 with tensorflow as backend

