import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

model = tf.keras.models.load_model('model.h5')

# Define the path to the test image directory and loop through all images in the directory
dir_path = 'E:\\X-ray classification\\xray_samples\\pract\\test_old\\err'
for i in os.listdir(dir_path):
    # Load each image using keras' load_img function, resize it to the required size, convert it to grayscale, and display it using matplotlib
    img = tf.keras.utils.load_img(dir_path+'//'+ i, target_size=(256, 256), color_mode='grayscale')
    plt.imshow(img, cmap='gray')
    plt.show()

    # Convert the image to a numpy array, expand its dimensions, and predict the class label (0 or 1) using the trained model
    X = np.expand_dims(img_to_array(img), axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    
    # Output a message indicating whether the image shows signs of pneumonia or not based on the predicted class label
    if val == 0:
        print("X-Ray shows signs of Pneumonia")
    else:
        print("X-Ray shows NO signs of Pneumonia")
