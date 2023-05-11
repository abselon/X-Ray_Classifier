# # from flask import Flask, request
# # from flask_cors import CORS
# # from PIL import Image

# # import tensorflow as tf
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import cv2
# # import os
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.optimizers import RMSprop
# # from tensorflow.keras.preprocessing.image import img_to_array
# # import pickle



# # app = Flask(__name__)
# # CORS(app)

# # @app.route('/')
# # def hello():
# #     return 'Hello, world! running on %s' % request.host

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the image from the request
# #     image_file =  request.files['imgfile']
# #     image = Image.open(image_file)
    
# #     # Pass the image to a Python function for processing
# #     prediction = process_image_function(image)
    
# #     # Return the processed image
# #     print(prediction)
# #     return {prediction}

# # def process_image_function(img):
# #     # Your image processing code goes here
# #     # For example, you could apply a filter to the image
# #     # and return the filtered image
# #     # load your model from a file
# #     with open('model.pkl', 'rb') as f:
# #         model = pickle.load(f)
# #     plt.imshow(img)
# #     plt.show()

# #     # Convert the image to a numpy array, expand its dimensions, and predict the class label (0 or 1) using the trained model
# #     X = np.expand_dims(img_to_array(img), axis = 0)
# #     images = np.vstack([X])
# #     val = model.predict(images)
    
# #     # Output a message indicating whether the image shows signs of pneumonia or not based on the predicted class label
# #     if val == 0:
# #         return ("X-Ray shows signs of Pneumonia")
# #     else:
# #         return ("X-Ray shows NO signs of Pneumonia")


# # if __name__ == '__main__':
# #     app.run(debug=True)

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# load your model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions on the input images
def predict(images):
    results = []
    for image in images:
        # Load the image using keras' load_img function, resize it to the required size, and convert it to a numpy array
        img = load_img(image, target_size=(200,200))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Predict the class label (0 or 1) using the trained model
        val = model.predict(x)

        # Store the result in a list
        if val == 0:
            results.append("shows NO signs of pneumonia")
        else:
            results.append("shows signs of pneumonia")
    return results

# Define the Streamlit app
def app():
    st.title("X-Ray Classifier")
    st.write("Upload X-Ray images to check if they show signs of pneumonia or not")

    # Allow the user to upload multiple images
    uploaded_files = st.file_uploader("Choose X-Ray images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # If images are uploaded, make predictions and display the results in batches
    if uploaded_files is not None:
        # Display the uploaded images and their corresponding results
        num_cols = 2
        num_images = len(uploaded_files)
        num_rows = int(np.ceil(num_images / num_cols))
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                index = i * num_cols + j
                if index < num_images:
                    image = np.array(load_img(uploaded_files[index]))
                    cols[j].image(image, caption='Uploaded X-Ray.', use_column_width=True)
                    result = predict([uploaded_files[index]])
                    cols[j].write(result[0])

# Run the Streamlit app
if __name__ == '__main__':
    app()



# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import pickle

# # load your model from a file
# model = tf.keras.models.load_model('model.h5')

# # # Load the saved model from the file
# # with open('model.pkl', 'rb') as f:
# #     model = pickle.load(f)

# # Define a function to make predictions on the input image
# # Define a function to make predictions on the input image
# def predict(image):
#     # Load the image using keras' load_img function, resize it to the required size, and convert it to a numpy array
#     img = load_img(image, target_size=(200, 200))
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     # Convert the numpy array to a tensor
#     x = tf.convert_to_tensor(x)

#     # Check the shape of the tensor and add an extra dimension with a size of 1 if necessary
#     if x.shape[-1] != 1:
#         x = tf.image.rgb_to_grayscale(x)
#     x = tf.expand_dims(x, axis=-1)

#     # Convert the grayscale image to an RGB image
#     x = tf.image.grayscale_to_rgb(x)

#     # Predict the class label (0 or 1) using the trained model
#     val = model.predict(x)

#     # Return the predicted class label as a string ("Pneumonia" or "No Pneumonia")
#     if val[0][0] == 0:
#         return "Pneumonia"
#     else:
#         return "No Pneumonia"


# # Define the Streamlit app
# def app():
#     st.title("X-Ray Classifier")
#     st.write("Upload one or more X-Ray images to check if they show signs of pneumonia or not")

#     # Allow the user to upload one or more images
#     uploaded_files = st.file_uploader("Choose one or more X-Ray images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

#     # If one or more images are uploaded, make a prediction and display the result for each image individually
#     if uploaded_files is not None:
#         # Display the images in batches of 4
#         for i in range(0, len(uploaded_files), 4):
#             batch = uploaded_files[i:i+4]
#             # Create a list to store the images and their predictions
#             images = []
#             # Loop through each image in the batch
#             for uploaded_file in batch:
#                 # Load the image and resize it to a smaller size
#                 image = np.array(load_img(uploaded_file, target_size=(200, 200)))
#                 # Make the prediction
#                 result = predict(uploaded_file)
#                 # Append the image and its prediction to the list
#                 images.append((image, result))
#             # Display the images and their predictions in a row
#             col1, col2, col3, col4 = st.columns(4)
#             for i, (image, result) in enumerate(images):
#                 if i == 0:
#                     col1.image(image, caption=result, use_column_width=True)
#                 elif i == 1:
#                     col2.image(image, caption=result, use_column_width=True)
#                 elif i == 2:
#                     col3.image(image, caption=result, use_column_width=True)
#                 elif i == 3:
#                     col4.image(image, caption=result, use_column_width=True)


# # Run the Streamlit app
# if __name__ == '__main__':
#     app()

