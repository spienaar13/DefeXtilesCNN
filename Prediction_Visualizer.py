# Prediction Visualizer
# Neural Netwrok Library
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

class testmodel:
    def __init__(self, path, batch, height, width):
      self.path = path
      self.batch = batch
      self.height = height
      self.width = width

      self.m = tf.keras.models.load_model(path)
      self.t = None
      self.tdir = None
      self.cm = None
      self.y_test = None
      self.y_pred = None
    
    def load_test_data(self, test_dir):
      datagen = ImageDataGenerator(rescale=1./255)
      data = datagen.flow_from_directory(
        test_dir,
        color_mode='grayscale',
        target_size=(self.height, self.width),
        batch_size=self.batch,
        class_mode='categorical',
        shuffle=False)
      self.t = data
      self.tdir = test_dir
    
    def eval(self):
       test_loss, test_acc = self.m.evaluate(self.t, verbose=0)
       print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

    def conf_matrix(self):
      predictions = np.array([])
      predictions = self.m.predict(self.t).round()
      pred_labels = np.argmax(predictions, axis=1)
      self.y_pred = pred_labels

      for _, y in self.t:
        labels = np.argmax(y, axis=1)
        break
      self.y_test = labels

      cm = confusion_matrix(labels, pred_labels)
      return cm
    
    def plt_cm(self, cm, title):
      classes = [name for name in sorted(os.listdir(self.tdir)) if os.path.isdir(os.path.join(self.tdir, name))]
      group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
      #group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
      #labels = [f'{v1}\n{v2}' for v1, v2 in
       #   zip(group_counts,group_percentages)]
      
      #for x in range(len(labels)):
      #  if(labels[x] == '0\n0.00%'): labels[x] = ''

      labels2 = np.asarray(group_counts).reshape(len(classes),len(classes))
    
      #cbar_kws={"labelsize": 16},
      plt.figure(figsize=(8,6), dpi=100)
      ax = sns.heatmap(cm/np.sum(cm), annot=labels2, fmt='', annot_kws={"fontsize": 16}, cmap='Blues')
      ax.set_xlabel("Predicted Class", fontsize=14, labelpad=5)
      ax.xaxis.set_ticklabels(classes)
      ax.set_ylabel("Actual Class", fontsize=14, labelpad=5)
      ax.yaxis.set_ticklabels(classes)
      ax.set_title(title, fontsize=16, pad=5)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.show()
    
    def report(self):
      print(classification_report(self.y_test, self.y_pred))


if __name__ == "__main__":
  image_height = 128
  image_width = 256
  batch_size = 96 #42
  model_path = './FINAL Models/LH_ModelFinal_f1_v100.h5'
  test_data = './LHdatasetAugX2'

  title = "Layer Height Prediction"

  X = testmodel(model_path, batch_size, image_height, image_width)
  X.load_test_data(test_data)
  X.eval()
  cm = X.conf_matrix()
  X.plt_cm(cm, title)
  X.report()





