# Neural Netwrok Library
import tensorflow as tf
#Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

epochs = 30
fold = 3
path = './Final Models'

training_accuracy = np.array([0])
validation_accuracy = np.array([0])


for root, dirs, files in os.walk(path):
    for filename in files:
            if filename.startswith('ND') :
                name = path+'/'+filename
                model = tf.keras.models.load_model(name)
                print(model.history.keys())
                training_accuracy = np.append(training_accuracy, model.history['accuracy'], axis=0)
                validation_accuracy = np.append(validation_accuracy, model.history['val_accuracy'], axis=0)


# Generating a lineplot illustrating these changes
epochs_array = np.array([0])
for i in range(fold-1):
    epochs_array = np.append(epochs_array, np.arange(1, epochs+1), axis = 0)

tdf = pd.DataFrame({'x': epochs_array, 'y': training_accuracy})
vdf = pd.DataFrame({'x': epochs_array, 'y': validation_accuracy})

sns.set_style("whitegrid")
sns.lineplot(data=tdf, x="x", y="y", color='blue', markers=True, dashes=False,ci='se', label="Training")
sns.lineplot(data =vdf, x="x", y="y", color='red', markers=True, dashes=False, ci='se', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Nozzle Speed Model')
plt.legend(title="Accuracy Type", loc='upper left')
plt.show()