
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
from time import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir_d = os.path.join(os.path.dirname(os.path.abspath("d_train.py")), "training_files_b")

train_img_d_generator = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 30,
                                   horizontal_flip = True,
                                   zoom_range = 0.2,
                                   shear_range = 0.3,
                                   validation_split = 0.2,  # Split dataset into training (80%) and validation (20%) dataset
                                   fill_mode = "nearest")

val_img_d_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2)  # Split dataset into training (80%) and validation (20%) dataset

train_img_d = train_img_d_generator.flow_from_directory(
    train_dir_d,
    subset="training",  # Set as training dataset
    target_size=(200,200),
    seed=123,
    class_mode="categorical")  # For multi-class classification purpose

val_img_d = val_img_d_generator.flow_from_directory(
    train_dir_d,
    subset="validation",  # Set as validation dataset
    target_size=(200,200),
    seed=123,
    class_mode="categorical")  # For multi-class classification purpose



# Learning process start time
start_time = time()

# Implementing Callbacks
threshold = 0.8
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') >= threshold): 
            self.model.stop_training = True
                        
callback_1 = myCallback()
callback_2 = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

# Model fitting
model_kem = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(200,200,3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation="softmax")
])

model_kem.summary()  # Review CNN architecture of the model

model_kem.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=0.001), metrics = ["accuracy"])

history_kem = model_kem.fit(
          train_img_d,
          epochs = 50,
          validation_data = val_img_d,
          callbacks = [callback_2],
          verbose = 2)

training_time = round((time()-start_time) / 60, 3)  # Count training time in minutes

print(f"Waktu Pelatihan: {training_time} menit")


# Saving Model
model_json_b = model_kem.to_json()
with open("model_b_kembangan.json", "w") as json_file:
    json_file.write(model_json_b)

model_kem.save_weights("model_b_kembangan.h5") 















