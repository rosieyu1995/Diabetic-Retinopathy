import os, sys
import numpy as np
from random import shuffle
import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping

np.random.seed(7)

#Set dimensions for our image
img_width, img_height = 224, 224


#Set base path for our project
base = '/Users/Howard/Documents/UMD_Document/Semester2/Big_Data/Project/data/train/'
base1 = '/Users/Howard/Documents/UMD_Document/Semester2/Big_Data/Project/data/validation/'


#Set epoch size to just one for now, since we are only interested in
# verifying the viability of the code for now rather than fully training our network
epochs = 20
batch_size = 32


# In[8]:

#Define the input_shape's dimensionality based on the image's data format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height) 
else:
    input_shape = (img_width, img_height, 3)


# #### Creating labels from images using a directory-based approach
# Diabetic retinopathy is graded on a clinical scale, which is numerical and corresponds to increasing severity of progression of the disease (0 – No DR, 1 – Mild, 2 – Moderate, 3 – Severe, 4 – Proliferative). Our training data contains images from all 5 classes, so we realized we would need to create a neural network capable of identifying between 5 different classes rather than a binary classification.

# In[9]:

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=30)
                                   #vertical_flip = True)
                                   #rotation_range=20,
                                   #brightness_range=[0.5, 1.5])#"An enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the original image."

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(base,
                                                    #shuffle= True,
                                                    #seed = True,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(base1,
                                            #shuffle = True,
                                            #seed = True,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[10]:

# look for the shape of the processing data
for data_batch, labels_batch in train_generator:
    print('data batch shape: ', data_batch.shape)
    print('labels batch shape: ', labels_batch.shape)
    break


# In[11]:

#Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=[224,224,3], kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(5, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('softmax'))
model.summary()


# In[12]:

len(train_generator)


# In[13]:

len(validation_generator)


# In[19]:

#Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(
        train_generator,
        steps_per_epoch=31615 // 32,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=3511 // 32,
        callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)])
