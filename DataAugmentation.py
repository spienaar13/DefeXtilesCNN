#DataAugmentation.py

# Helper Libraries
import tensorflow as tf
import os 


# Helper Functions:
# Random uniforming cropping of data finction- for data augmentation purposes
# Minimum unit of the random crop is equal to 200x900pixels
CROP_HEIGHT, CROP_WIDTH = 400, 1800
def random_crop(image):
  cropped_image = tf.image.random_crop(image, size=[CROP_HEIGHT, CROP_WIDTH, 1])
  return cropped_image

# Image resizing
IMG_height, IMG_width  = 128, 256
def image_resize(image):
    resized = tf.image.resize(image, (IMG_height, IMG_width))
    return resized

class Augmenter:
    # Specs for Augmenter Class of a specific directory
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = [name for name in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, name))]

    # Function to augment the data in at given location with specific number of crops
    def Data_Aug(self, crops):
        for i in range(len(self.classes)):
            path = self.root_dir + self.classes[i]
            newpath = self.root_dir[:-1] + 'AugX'+ str(crops) + '/' + self.classes[i]
            # Making a new directory if it doesn't already exist
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            # Reading into grayscaling, randomly cropping and resizing images before writing to new directory
            for image in sorted(os.listdir(path)):
                j = 1
                if image.endswith(".jpg"):
                    file = path + "/" + image
                    im1 = tf.io.read_file(file)
                    im1 = tf.image.decode_jpeg(im1, channels=1)
            
                    while  j < (int(crops)+1):
                        newname = newpath + "/" + str(j)+ "_" + image
                        im2 = random_crop(im1)
                        im2 = image_resize(im2)
                        im2 = tf.cast(im2, tf.uint8)
                        im3 = tf.image.encode_jpeg(im2)
                        tf.io.write_file(newname, im3)
                        j = j+1
            i = i+1 
        return print('DataAugmentation Complete!')


if __name__ == "__main__":
    # Directory
    path = "./NDdataset/"

    aug = Augmenter(path)
    aug.Data_Aug(2)
