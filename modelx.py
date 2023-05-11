# Import required libraries
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
# Define train and validation ImageDataGenerator objects with rescaling factor of 1/255
train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

# Load training and validation data from the given directories and specify the target image size and batch size
train_dataset = train.flow_from_directory("E:\\X-ray classification\\xray_samples\\pract\\train", target_size= (200, 200), batch_size = 4, class_mode = 'binary')
validation_dataset = train.flow_from_directory("E:\\X-ray classification\\xray_samples\\validation", target_size= (200, 200), batch_size = 4, class_mode = 'binary')

# Check the class indices for the two classes (pneumonia and non-pneumonia)
train_dataset.class_indices

# Define the model architecture as a Sequential object with multiple layers of convolution and pooling, followed by two fully connected layers
model =tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape = (200, 200, 3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (200, 200, 3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (200, 200, 3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Flatten(),
                                   #
                                   tf.keras.layers.Dense(512, activation = 'relu'),
                                   #
                                   tf.keras.layers.Dense(1,activation='sigmoid')
                                   ])

# Compile the model by specifying the loss function, optimizer, and metrics to be used during training
model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(learning_rate = 0.001),
              metrics = ['accuracy']
              )

# Train the model on the training data using the fit method and specifying the number of steps per epoch, number of epochs, and validation data
model_fit = model.fit(train_dataset,
                      steps_per_epoch = 10,
                      epochs = 300,
                      validation_data = validation_dataset)

# save your model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)