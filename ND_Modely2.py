# Neural Netwrok Library
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define global variables
data_dir = "/Users/spienaar/Desktop/Deformations Research/ML Model/NDdatasetAugX14/"
image_height = 128
image_width = 256
batch_size = 672
epochs=30

# Load and preprocess the image dataset from the local directory
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    data = datagen.flow_from_directory(
        data_dir,
        color_mode='grayscale',
        target_size=(image_height, image_width),
        batch_size=batch_size, 
        class_mode='categorical',
        shuffle=True,
        seed=42 )
    return data

# Function to train create model
def create_model():
    model = Sequential([
        #layers.Rescaling(1./255, input_shape=(image_height, image_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
        ])
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
    
# Define the early stopping callback and 
#early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Load data 
data = load_data(data_dir)

num_classes = len(data.class_indices)
X, y = data.next()

# Define the number of folds for k-fold cross validation and create object
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle = True)

# Initialize variables for performance metrics
accuracies = []
loss = []
training_accuracy = np.array([0])
validation_accuracy = np.array([0])
fold = 1

# Train and validate the model using K-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create a new model for each fold
    model = create_model()

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)

    # Store accuracies for graphing
    training_accuracy = np.append(training_accuracy, history.history['accuracy'], axis=0)
    validation_accuracy = np.append(validation_accuracy, history.history['val_accuracy'], axis=0)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'Fold {fold} - Validation loss: {val_loss}, Validation accuracy: {val_acc}')
    
    accuracies.append(val_acc)
    loss.append(val_loss)
    model.save(str('ND_ModelFinal'+'_f'+str(fold)+'_v'+ str(round(val_acc*100)) +'.h5'))
    fold += 1

# Calculate the average accuracy and loss
avg_accuracy = np.mean(accuracies)
acc_err = np.std(accuracies) / np.sqrt(len(accuracies))
print(f'Average Validation Accuracy: {avg_accuracy} +- {acc_err}')
avg_loss = np.mean(loss)
loss_err = np.std(loss) / np.sqrt(len(loss))
print(f'Average Validation Loss: {avg_loss} +-  {loss_err}')

# Generating a lineplot illustrating these changes
epochs_array = np.array([0])
for i in range(fold-1):
    epochs_array = np.append(epochs_array, np.arange(1, epochs+1), axis = 0)

tdf = pd.DataFrame({'x': epochs_array, 'y': training_accuracy})
vdf = pd.DataFrame({'x': epochs_array, 'y': validation_accuracy})

sns.set_style("whitegrid")
sns.lineplot(data=tdf, x="x", y="y", color='blue', markers=True, dashes=False, errorbar = ('ci', 68), label="Training")
sns.lineplot(data =vdf, x="x", y="y", color='red', markers=True, dashes=False, errorbar = ('ci', 68), label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Nozzle Diameter Model')
plt.legend(title="Accuracy Type", loc='upper left')
plt.show()





