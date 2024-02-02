#playground.py
import tensorflow as tf
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.activation_maximization.callbacks import Progress
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)

    # Visualize the specific layer's via their
    def filterVis(self, layer_name, names, title):
        length =  len(names)
        class_number = [None]*length

        for count, _ in enumerate(names):
            class_number[count] = count


        #Used to constructs new model whose output is replaced to layer name's output.
        extract_intermediate_layer = ExtractIntermediateLayer(index_or_name=layer_name)
        # Used to replace softmax activation with linear activation
        replace2linear = ReplaceToLinear()

        #Score function that returns the target scores of the classes
        score = CategoricalScore(class_number)
        
        # Noisey geneeric input to be maximized
        seed_input = tf.random.uniform((length, 128, 256, 1), 0, 255)
        # Defining the activation maximazation instances with out desired alterations
        activation_maximization = ActivationMaximization(self.model,
                            model_modifier=[extract_intermediate_layer, replace2linear],
                            clone=False)
        # Runnning activation maximzation on our seed_input using out score
        activations = activation_maximization(score, seed_input=seed_input,callbacks=[Progress()])
        
        #Plot the images of the activation
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
        f.suptitle(title, fontsize=20)
        for i in range(3):
            ax[i].set_title(names[i], fontsize=16)
            ax[i].imshow(activations[i])
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    LH_layer_path = './FINAL Models/LH_ModelFinal_f2_v100.h5'
    ND_layer_path = './FINAL Models/ND_ModelFinal_f1_v98.h5'
    EM_layer_path = './FINAL Models/EM_ModelFinal_f1_v100.h5'
    NS_layer_path = './FINAL Models/NS_ModelFinal_f1_v98.h5'

    EM_layer_name = 'dense_1'
    LH_layer_name = 'dense_3'
    NS_layer_name = 'dense_1'
    ND_layer_name = 'dense_1'

    ND_names = ['0.4', '0.6', '0.8', '1.0']
    EM_names = ['0.3', '0.4', '0.5', '0.6']
    LH_names = ['26', '35', '44']
    NS_names = ['22.5', '45', '60']

    vis = Visualizer(NS_layer_path)
    vis.filterVis(NS_layer_name, NS_names, 'Nozzle Speed Classes')