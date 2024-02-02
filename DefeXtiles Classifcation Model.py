# DefeXtiles Classifcation Model
# Written by Stephane Pienaar

#
# Aims to classify the Nozzle Diameter, 
# Date: 
# 

# Neural Netwrok Library
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensforflow.keras.preprocessing import image_dataset_from_directory

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import os 


# Helper Functions:
# Random uniforming cropping of data finction- for data augmentation purposes
# Minimum unit of the random crop is equal to 900x200pixels
CROP_HEIGHT, CROP_WIDTH = 200, 900
def random_crop(image):
  cropped_image = tf.image.random_crop(image, size=[CROP_HEIGHT, CROP_WIDTH, 1])
  return cropped_image

# Image resizing
IMG_SIZE = 100
def image_resize(image):
    resized = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return resized

# Convert to GeryScale
def image_gray(image):
    gray = tf.image.rgb_to_grayscale(image)
    return gray

# Loading dataset:
folders = ["N1", "N4", "N6", "N8"]
scans  = []
for i in range(len(folders)):
# for i in range(len(dataset)):
    path = "/Users/spienaar/Desktop/Deformations Research/ML Model/Dataset1/" + folders[i]
    for image in sorted(os.listdir(path)):
        if image.endswith(".jpg"):
            file = path + "/" + image
            image = tf.io.read_file(file)
            image = tf.image.decode_jpeg(image, channels=1)
            scans.append(image)
    i = i+1 



# Inputing labels for dataset
# Label format: ['nozzle Diameter', 'Layer Height (mm)', 'Extrusion Multiplier', 'velocity (mm/s)']
# 1mm Nozzle Diameter Labels:
N1001_label = [1, 0.6565, 0.4, 15] 
N1002_label = [1, 0.65325, 0.6, 15] 
N1003_label = [1, 0.65325, 0.6, 15] 
N1004_label = [1, 0.65325, 0.5, 30]
N1005_label = [1, 0.4742635, 0.4, 40] 
N1006_label = [1, 0.8, 0.8, 30]

N1008_label = [1, 0.8, 0.6, 30]
N1009_label = [1, 0.8, 0.7, 30]

N1010_label = [1, 0.4742635, 0.3, 30]
N1012_label = [1, 0.4742635, 0.5, 30]
N1013_label = [1, 0.65325, 0.5, 60]
N1014_label = [1, 0.65325, 0.65, 15]

# 0.4mm Nozzle Diamter Labels:
N4001_label =  [0.4, 0.15, 0.25, 60]  
N4002_label =  [0.4, 0.15, 0.3, 60]   
#N4008_label =  [0.4, 0.15, 0.35, 60]  
N4003_label =  [0.4, 0.15, 0.4,  60]   
N4011_label =  [0.4, 0.15, 0.45, 60]   
N4007_label =  [0.4, 0.15, 0.5, 60]  

#N4013_label =  [0.4, 0.11, 0.3, 60]   
N4005_label =  [0.4, 0.11, 0.4,  60]   
N4006_label =  [0.4, 0.11, 0.45, 60]   

N4009_label =  [0.4, 0.15, 0.4, 120]   
N4014_label =  [0.4, 0.15, 0.4,  30]   
#N4012_label =  [0.4, 0.15, 0.45,120]   
#N4010_label =  [0.4, 0.15, 0.5, 120]  

N4015_label = [0.4, 0.26, 0.65, 60]  
N4016_label = [0.4, 0.26, 0.6, 60]  
#N4017_label = [0.4, 0.261, 0.55, 60]
N4018_label = [0.4, 0.26, 0.5, 60]  

# 0.6mm Nozzle Diamter Labels:
N6003_label = [0.6, 0.35, 0.3, 45] 
N6006_label = [0.6, 0.35, 0.4, 45]  
N6001_label = [0.6, 0.35, 0.5, 45]
N6004_label = [0.6, 0.35, 0.6, 45]
N6008_label = [0.6, 0.35, 0.5, 22.5] # same as N60031_label
N6025_label = [0.6, 0.35, 0.4, 60]
N6026_label = [0.6, 0.35, 0.5, 60] # same as N60032_label
N6027_label = [0.6, 0.35, 0.6, 60]

N6009_label = [0.6, 0.44, 0.5, 45] 
N6011_label = [0.6, 0.44, 0.6, 45]
N6028_label = [0.6, 0.44, 0.6, 60] 
N6029_label = [0.6, 0.44, 0.5, 60]
N6015_label = [0.6, 0.44, 0.6, 22.5]
N6018_label = [0.6, 0.44, 0.5, 22.5]

N6012_label = [0.6, 0.26, 0.3, 45] 
N6014_label = [0.6, 0.26, 0.4, 45] 
N6013_label = [0.6, 0.26, 0.5, 45]
N6019_label = [0.6, 0.26, 0.4, 22.5]
N6022_label = [0.6, 0.26, 0.3, 22.5] # w/ duplicate N6030_label



##N6007_label = [0.6, 0.35, 0.5, 90]
##N6010_label = [0.6, 0.44, 0.7, 45]

##N6024_label = [0.6, 0.35, 0.6, 90]
##N6016_label = [0.6, 0.44, 0.6, 90]
##N6017_label = [0.6, 0.44, 0.5, 90]


# 0.8mm Nozzle Diamter Labels:
N8002_label = [.8, 0.3, 0.2, 45]
N8007_label = [.8, 0.3, 0.3, 45]
N8006_label = [.8, 0.3, 0.4, 45]

N8008_label = [.8, 0.4, 0.2, 45]
N8010_label = [.8, 0.4, 0.3, 45]
N8012_label = [.8, 0.4, 0.4, 45]
N8009_label = [.8, 0.4, 0.5, 45]

N8014_label = [0.8, 0.4, 0.4, 17.5]
N8013_label = [0.8, 0.4, 0.4, 70]
N8011_label = [0.8, 0.4, 0.4, 100]

N8004_label = [.8, 0.5, 0.4, 45]
N8003_label = [.8, 0.5, 0.5, 45]
N8001_label = [.8, 0.5, 0.6, 45]

labels = np.array([N1001_label, N1002_label, N1003_label, N1004_label, N1005_label, N1006_label, N1008_label, N1009_label, N1010_label, N1012_label, N1013_label, N1014_label, N4001_label, N4002_label, N4003_label, N4005_label, N4006_label, N4007_label, N4008_label, N4009_label, N4010_label, N4011_label, N4012_label, N4013_label, N4014_label, N4015_label, N4016_label, N4017_label, N4018_label, N6001_label, N6003_label, N6004_label, N6006_label,  N6007_label, N6008_label, N6009_label, N6010_label, N6011_label, N6012_label, N6013_label, N6014_label, N8001_label, N8002_label, N8003_label, N8004_label, N8006_label, N8007_label, N8008_label, N8009_label, N8010_label, N8011_label, N8012_label, N8013_label, N8014_label])

ND_labels = labels[:, 0]
ND_Dataset = tf.data.Dataset.from_tensor_slices(ND_labels)
#LH_labels = labels[:, 1]
#EM_labels = labels[:, 2]
#NV_labels = labels[:, 3]

# Create a list of file paths to the images
folders = ["N1", "N4", "N6", "N8"]
scans  = []
for i in range(len(folders)):
# for i in range(len(dataset)):
    path = "/Users/spienaar/Desktop/Deformations Research/ML Model/Dataset1/" + folders[i]
    for image in sorted(os.listdir(path)):
        if image.endswith(".jpg"):
            scans.append(image)

ND_Dataset = tf.data.Dataset.from_tensor_slices(scans)

for element in ND_Dataset:
    print(element)

# Define a function to load the images and return the image data and labels
def load_image(file_path):
    # Load the image using TensorFlow
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = random_crop(image)
    image = tf.image.resize(image, [256, 256])
    return image

# Use the map function to apply the load_image function to each file path
ND_Dataset = ND_Dataset.map(load_image)

# Print the first batch of images
for images in ND_Dataset:
    print(images)


ND_model = Sequential([
  layers.Rescaling(1./255, input_shape=(CROP_HEIGHT, CROP_WIDTH, 1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(ND_classes)
])

ND_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

ND_model.summary()

ND_model.fit(train_ds, 
             epochs = 5,  
             verbose = 1, 
             validation_data = val_ds,
             shuffle=True)

