#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install tensorflow-datasets')


# In[3]:


import shutil 
import pandas as pd


# In[7]:


import os
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import glob
import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn import svm
from PIL import Image
import random

import tensorflow_datasets as tfds
from tensorflow.keras import layers


# In[8]:


import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform


# In[9]:


train_dataset = '/Users/ericyao/Documents/AARD/Lung/archive/raw_png/Training'
testing_dataset = '/Users/ericyao/Documents/AARD/Lung/archive/raw_png/Testing'


# In[10]:


img_paths = glob.glob(os.path.join(train_dataset, '*/*.png'))
parent_names = [os.path.basename(os.path.abspath(os.path.join(p, os.pardir))) for p in img_paths]
labels = np.asarray([0 if p == '0' else 1 if p == '1' else 2 if p == '2' 
                    else 3 for p in parent_names])
imgs = np.asarray([imread(p) for p in img_paths])
len(img_paths)


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.25, random_state = 1)


# In[12]:


y_test


# In[13]:


train_datagen = ImageDataGenerator(rotation_range = 20)
test_datagen = ImageDataGenerator()


# In[14]:


trdata = ImageDataGenerator(rotation_range=20)
traindata = trdata.flow_from_directory("/Users/ericyao/Documents/AARD/Lung/archive/raw_png/Training",target_size=(224,224))
tsdata = ImageDataGenerator(rotation_range=20)
testdata = tsdata.flow_from_directory("/Users/ericyao/Documents/AARD/Lung/archive/raw_png/Testing", target_size=(224,224))


# In[15]:


from tensorflow.keras.applications.resnet import ResNet50


# In[16]:


model = tf.keras.models.Sequential([
    ResNet50(weights='imagenet', input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4,activation = "softmax")  
])


# In[17]:


model.summary()


# In[18]:


from tensorflow.keras.optimizers import Adam


# In[19]:


from tensorflow.keras.optimizers import RMSprop,SGD,Adam
adam=Adam(lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])


# In[20]:


es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)


# In[21]:


mc = ModelCheckpoint('Resnet.h5', monitor='val_accuracy', mode='acc')


# In[22]:


history = model.fit_generator(steps_per_epoch=100, generator=traindata,validation_data= testdata,  validation_steps= 10,epochs=1, callbacks=[mc, es])  


# In[23]:


scores = model.predict(testdata, verbose = 1)


# In[24]:


model.evaluate(testdata, verbose = 1)


# In[25]:


preds = np.argmax(scores, axis = 1)


# In[26]:


preds


# In[27]:


matrix = confusion_matrix(testdata.classes, preds)
disp = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = traindata.class_indices)
disp = disp.plot(cmap = plt.cm.Blues)
plt.xticks(rotation = 90)
plt.show()


# In[28]:


print(matrix)


# In[29]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[0].set(xlim=(0, 100), ylim=(0, 2))
    ax[0].set_xlabel('Epochs')
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[1].set(xlim=(0, 100), ylim=(0, 1))
    ax[1].set_xlabel('Epochs')
    ax[0].legend()
    ax[1].legend()


# In[30]:


show_final_history(history)


# In[ ]:




