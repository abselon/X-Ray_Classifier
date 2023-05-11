from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import cv2
import numpy as np

def load_test_data():
    # Set the path to your test data directory
    test_dir = 'E:\\X-ray classification\\xray_samples\\pract\\test_old'

    # Get a list of all image filenames in the test data directory
    test_filenames = os.listdir(test_dir)

    # Initialize empty lists for the images and labels
    images = []
    labels = []

    # Loop over each image filename in the test data directory
    for filename in test_filenames:
        # Load the image and convert it to grayscale
        img = cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Resize the image to a fixed size (e.g. 256x256)
        img = cv2.resize(img, (256, 256))

        # Convert the image to a numpy array and normalize its values to [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0

        # Add the image to the list of images
        images.append(img)

        # Extract the label from the filename (e.g. 'pneumonia' or 'normal')
        label = filename.split('.')[0]
        labels.append(label)

    # Convert the list of images and labels to numpy arrays
    x_test = np.array(images)
    y_test = np.array(labels)

    return x_test, y_test


# Load your trained model and test data
model = load_model('model.h5')
x_test, y_test = load_test_data()

# Generate predicted probabilities for the test data
y_pred = model.predict(x_test)

# Calculate the FPR, TPR, and thresholds for the predicted probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
