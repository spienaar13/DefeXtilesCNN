import tensorflow as tf
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.activation_maximization.callbacks import Progress
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16 as Model


class Visualizer:
    def __init__(self):
        self.model = Model(weights='imagenet', include_top=True)

    # Visualize the specific layer's via their
    def filterVis(self, label_number, names):
        length =  len(names)


        # Used to replace softmax activation with linear activation
        replace2linear = ReplaceToLinear()

        #Score function that returns the target scores of the classes
        score = CategoricalScore(label_number)
        # Noisey geneeric input to be maximized
        seed_input = tf.random.uniform((length, 224, 224, 3), 0, 255)

        # Defining the activation maximazation instances with out desired alterations
        activation_maximization = ActivationMaximization(self.model,
                            model_modifier=replace2linear,
                            clone=True)
        
        # Runnning activation maximzation on our seed_input using out score
        activations = activation_maximization(score, seed_input=seed_input,callbacks=[Progress()])
        
        #Plot the images of the activation
        f, ax = plt.subplots(nrows=1, ncols=length, figsize=(14, 5))

        for i in range(length):
            title  = 'Class: ' + names[i]
            ax[i].set_title(title, fontsize=12)
            ax[i].imshow(activations[i])
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    label_number = [4, 108, 146 ]

    names = ['Hammerhead Shark', 'JellyFish', 'Pelican']

    vis = Visualizer()
    vis.filterVis(label_number, names)