
'''
Menguji hasil latihan dengan data set A; metode standar

REGU 3 : 

- Muhammad Satrio Athiffardi Prasiddha (13318032) 

- Tommy Akbar Taufik (13319044) 

- Arsyi Adlani(13319105) 

'''

print('PENGENAL CORAK BATIK')
print('Sampel: Dataset A')
print('Metode: Kembangan (Convolutional Neural Networks')
print('------')

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from keras.models import model_from_json



# Loading Model
json_file = open('model_b_kembangan.json', 'r')
loaded_model_json_d_kembangan = json_file.read()
json_file.close()
loaded_model_d_kembangan = model_from_json(loaded_model_json_d_kembangan)


# load weights into new model
loaded_model_d_kembangan.load_weights("model_b_kembangan.h5")

folder_path = os.path.join(os.path.dirname(os.path.abspath("d_action.py")), "training_files_b")

train_dir_d_kembangan = folder_path

train_img_d_kembangan_generator = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 30,
                                   horizontal_flip = True,
                                   zoom_range = 0.2,
                                   shear_range = 0.3,
                                   validation_split = 0.2,  # Split dataset into training (80%) and validation (20%) dataset
                                   fill_mode = "nearest")

val_img_d_kembangan_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2)  # Split dataset into training (80%) and validation (20%) dataset

train_img_d_kembangan = train_img_d_kembangan_generator.flow_from_directory(
    train_dir_d_kembangan,
    subset="training",  # Set as training dataset
    target_size=(200,200),
    batch_size=32,
    shuffle=False,
    class_mode="categorical")  # For multi-class classification purpose

val_img_d_kembangan= val_img_d_kembangan_generator.flow_from_directory(
    train_dir_d_kembangan,
    subset="validation",  # Set as validation dataset
    target_size=(200,200),
    batch_size=32,
    shuffle=False,
    class_mode="categorical")  # For multi-class classification purpose


test_path= os.path.join(os.path.dirname(os.path.abspath("d_action.py")), "test_files")

test_dir_d_kembangan = test_path

i = 1

folder_path_test = os.path.join(os.path.dirname(os.path.abspath("d_action.py")), "test_files")
train_classes =list(train_img_d_kembangan.class_indices.keys())

for file_name in os.listdir(folder_path_test):
  file_path_test = os.path.join(folder_path_test, file_name)
  uploaded_img = image.load_img(file_path_test, target_size=[200, 200])
  img_array = image.img_to_array(uploaded_img)
  img_array = img_array/255
  img_array = np.expand_dims(img_array, axis=0)
  img_array = np.vstack([img_array])
  cls = loaded_model_d_kembangan.predict(img_array)

  print(f'Data Test ke {i}: {file_name}')
  plt.imshow(uploaded_img)
  plt.show()

  print(train_classes)
  print(cls)
  print()
  i += 1

print('------')


### Confusion Matrix Functions

def true_positive(cm):
  kawung = 0
  mega = 0
  parang = 0
  truntum = 0
  for i in range(4):
    for j in range(4):
      if i==0 and j==0:
        kawung += cm[i][j]
      if i==1 and j==1:
        mega += cm[i][j]
      if i==2 and j==2:
        parang += cm[i][j]
      if i==3 and j==3:
        truntum += cm[i][j]
  return [kawung, mega, parang, truntum]

def true_negative(cm):
  kawung = 0
  mega = 0
  parang = 0
  truntum = 0
  for i in range(4):
    for j in range(4):
      if i!=0 and j!=0:
        kawung += cm[i][j]
      if i!=1 and j!=1:
        mega += cm[i][j]
      if i!=2 and j!=2:
        parang += cm[i][j]
      if i!=3 and j!=3:
        truntum += cm[i][j]
  return [kawung, mega, parang, truntum]

def false_positive(cm):
  kawung = 0
  mega = 0
  parang = 0
  truntum = 0
  for i in range(4):
    for j in range(4):
      if i==0 and j!=0:
        kawung += cm[i][j]
      if i==1 and j!=1:
        mega += cm[i][j]
      if i==2 and j!=2:
        parang += cm[i][j]
      if i==3 and j!=3:
        truntum += cm[i][j]
  return [kawung, mega, parang, truntum]

def false_negative(cm):
  kawung = 0
  mega = 0
  parang = 0
  truntum = 0
  for i in range(4):
    for j in range(4):
      if i!=0 and j==0:
        kawung += cm[i][j]
      if i!=1 and j==1:
        mega += cm[i][j]
      if i!=2 and j==2:
        parang += cm[i][j]
      if i!=3 and j==3:
        truntum += cm[i][j]
  return [kawung, mega, parang, truntum]

def evaluate(cm):
  matriks = pd.DataFrame([true_positive(cm), 
             true_negative(cm),
             false_positive(cm),
             false_negative(cm)])

  matriks.columns = ['Kawung', 'Mega', 'Parang', 'Truntum']
  matriks['TOTAL'] = matriks['Kawung'] + matriks['Mega'] + matriks['Parang'] + matriks['Truntum']
  matriks['Kinerja'] = ['True-Positif', 'True-Negatif', 'False-Positif', 'False-Negatif']
  matriks = matriks.set_index('Kinerja')

  accuracy = (matriks['TOTAL'].loc['True-Positif'] + matriks['TOTAL'].loc['True-Negatif']) / np.sum(matriks['TOTAL'])
  precision = (matriks['TOTAL'].loc['True-Positif']) / ((matriks['TOTAL'].loc['True-Positif'] + matriks['TOTAL'].loc['False-Positif']))
  recall = (matriks['TOTAL'].loc['True-Positif']) / ((matriks['TOTAL'].loc['True-Positif'] + matriks['TOTAL'].loc['False-Negatif']))
  selectivity = (matriks['TOTAL'].loc['True-Negatif']) / (matriks['TOTAL'].loc['True-Negatif'] + matriks['TOTAL'].loc['False-Positif'])
  f1_score = (2*precision*recall) / (precision+recall)

  print(matriks)
  print('------')
  print(f'Accuracy: {accuracy*100:.2f}%')
  print(f'Precision: {precision*100:.2f}%')
  print(f'Selectivity: {selectivity*100:.2f}%') 
  print(f'F1-Score: {f1_score*100:.2f}%')

print('Kinerja Proses Training')
print()
train_pred_d_kembangan = loaded_model_d_kembangan.predict(train_img_d_kembangan)
train_predictions_d_kembangan = np.argmax(train_pred_d_kembangan, axis=1)

cm_train_d_kembangan = metrics.confusion_matrix(train_predictions_d_kembangan, train_img_d_kembangan.classes)

evaluate(cm_train_d_kembangan)
print()

print('------')
print('Kinerja Proses Validation')
print()
val_pred_d_kembangan = loaded_model_d_kembangan.predict(val_img_d_kembangan)
val_predictions_d_kembangan = np.argmax(val_pred_d_kembangan, axis=1)

cm_val_d_kembangan = metrics.confusion_matrix(val_predictions_d_kembangan, val_img_d_kembangan.classes)

evaluate(cm_val_d_kembangan)

