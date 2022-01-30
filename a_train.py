
'''
Melatih model dengan data set A; metode standar

REGU 3 : 

- Muhammad Satrio Athiffardi Prasiddha (13318032) 

- Tommy Akbar Taufik (13319044) 

- Arsyi Adlani(13319105) 

'''


# Mengimport library yang dibutuhkan
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers
from sklearn import metrics
from tensorflow.keras import Model
from sklearn.metrics import classification_report
from time import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### TRAINING DATASET A

train_dir_a_incep = os.path.join(os.path.dirname(os.path.abspath("a_train.py")), "training_files_a")

train_img_a_incep_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2)  # Split dataset into training (80%) and validation (20%) dataset

val_img_a_incep_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2)  # Split dataset into training (80%) and validation (20%) dataset

train_img_a_incep = train_img_a_incep_generator.flow_from_directory(
    train_dir_a_incep,
    subset="training",  # Set as training dataset
    target_size=(200,200),
    batch_size=32,
    shuffle=False,
    class_mode="categorical")  # For multi-class classification purpose

val_img_a_incep = val_img_a_incep_generator.flow_from_directory(
    train_dir_a_incep,
    subset="validation",  # Set as validation dataset
    target_size=(200,200),
    batch_size=32,
    shuffle=False,
    class_mode="categorical")  # For multi-class classification purpose

#LOAD MODUL INSEPTION-V3

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model_a_incep = InceptionV3(input_shape = (200, 200, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model_a_incep.load_weights(local_weights_file)

for layer in pre_trained_model_a_incep.layers:
    layer.trainable = False

last_layer = pre_trained_model_a_incep.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


#MODELLING

# Learning process start time
start_time = time()

# Implementing Callbacks
threshold = 0.8
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') >= threshold): 
            self.model.stop_training = True
                        
callback_1 = myCallback()
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Model fitting

x = layers.Flatten()(last_output)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (4, activation='softmax')(x)        


model_a_incep = Model( pre_trained_model_a_incep.input, x) 


model_a_incep.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=0.001), metrics = ["accuracy"])

history_a_incep = model_a_incep.fit(
          train_img_a_incep,
          epochs = 50,
          validation_data = val_img_a_incep,
          callbacks = [callback_2],
          verbose = 2)

training_time = round((time()-start_time) / 60, 3)  # Count training time in minutes

print(f"Waktu Pelatihan: {training_time} menit")



### Loss Visualization

"""Plot training and validation loss during learning process."""

epochs = list(range(1, len(history_a_incep.history["loss"])+1))

plt.plot(epochs, history_a_incep.history["loss"], marker = "o", label = "Training Data")  # Plot training loss
plt.plot(epochs, history_a_incep.history["val_loss"], marker = "o", label = "Validation Data")  # Plot validation loss

plt.title("LOSS GRAPH", weight="bold")
plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(b=1, axis="y")  # Show graph grid
plt.legend()

plt.show()


### Accuracy Visualization


"""Plot training and validation accuracy during learning process."""

import matplotlib.pyplot as plt

epochs = list(range(1, len(history_a_incep.history["accuracy"])+1))

plt.plot(epochs, history_a_incep.history["accuracy"], marker = "o", label = "Training Data")  # Plot training accuracy
plt.plot(epochs, history_a_incep.history["val_accuracy"], marker = "o", label = "Validation Data")  # Plot validation accuracy

plt.title("ACCURACY GRAPH", weight="bold")
plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(b=1, axis="y")  # Show graph grid
plt.legend()

plt.show()

print()

## Confusion Matrix of Training Process
print("Confusion Matrix of Training Process")

train_class_labels_a_incep = list(train_img_a_incep.class_indices.keys())
train_class_labels_a_incep

train_pred_a_incep = model_a_incep.predict(train_img_a_incep)
train_predictions_a_incep = np.argmax(train_pred_a_incep, axis=1)

cm_train_a_incep = metrics.confusion_matrix(train_predictions_a_incep, train_img_a_incep.classes)
cr_train_a_incep = classification_report(train_img_a_incep.classes, train_predictions_a_incep, target_names=train_class_labels_a_incep)

print(cm_train_a_incep)
print()
print(cr_train_a_incep)
heatmap_train=sns.heatmap(cm_train_a_incep, annot=True)
print(heatmap_train)

print()

## Confusion Matrix of Validation Process
print("Confusion Matrix of Validation Process")

val_class_labels_a_incep = list(val_img_a_incep.class_indices.keys())
val_class_labels_a_incep

val_pred_a_incep = model_a_incep.predict(val_img_a_incep)
val_predictions_a_incep = np.argmax(val_pred_a_incep, axis=1)

cm_val_a_incep = metrics.confusion_matrix(val_predictions_a_incep, val_img_a_incep.classes)
cr_val_a_incep = classification_report(val_img_a_incep.classes, val_predictions_a_incep, target_names=val_class_labels_a_incep)

print(cm_val_a_incep)
print()
print(cr_val_a_incep)
heatmap_val = sns.heatmap(cm_val_a_incep, annot=True)
print(heatmap_val)

# Saving Model
model_json_a_incep = model_a_incep.to_json()
with open("model_a_inception.json", "w") as json_file:
    json_file.write(model_json_a_incep)

model_a_incep.save_weights("model_a_incep_inception.h5")