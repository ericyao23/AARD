#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install tensorflow-datasets')


# In[3]:


import shutil 
import os
import pandas as pd


# In[4]:


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


# In[32]:


train_dataset = '/Users/ericyao/Documents/AARD/Lung/archive/raw2_png/Training'
testing_dataset = '/Users/ericyao/Documents/AARD/Lung/archive/raw2_png/Testing'


# In[33]:


data= pd.read_csv("metadata3.csv")
data


# In[34]:


img_paths = glob.glob(os.path.join(train_dataset, '*/*.png'))
parent_names = [os.path.basename(os.path.abspath(os.path.join(p, os.pardir))) for p in img_paths]
labels = np.asarray([1 if p == '1' else 2 if p == '2' 
                    else 3 for p in parent_names])
imgs = np.asarray([imread(p) for p in img_paths])
len(img_paths)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.25, random_state = 1)


# In[36]:


y_test


# In[62]:


import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (512,512,3)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(3,activation = "softmax")  
])


# In[66]:


model = tf.keras.Model(x_in, x_out)
model.summary()

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
adam=Adam(lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
# In[69]:


bs=32       
train_dir = "/Users/ericyao/Documents/AARD/Lung/archive/raw2_png/Training" 
validation_dir = "/Users/ericyao/Documents/AARD/Lung/archive/raw2_png/Testing"   
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator( 
    rotation_range=20,
    brightness_range=[0.2,1.0],
    featurewise_center=True,
    featurewise_std_normalization=True,
    )

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs,class_mode='categorical',target_size=(512,512))

validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=bs,
                                                         class_mode  = 'categorical',
                                                         target_size=(512,512))


# In[70]:


import math


# In[71]:


compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / 32))

steps_per_epoch = compute_steps_per_epoch(90)
val_steps = compute_steps_per_epoch(60)


# In[73]:


from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[74]:


es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)


# In[75]:


mc = ModelCheckpoint('Resnet.h5', monitor='val_accuracy', mode='acc')


# In[76]:


history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=60,
                    validation_steps= val_steps,
                    verbose=2,
                    callbacks=[mc, es])


# In[54]:


scores = model.predict(validation_generator, verbose = 1)


# In[55]:


model.evaluate(validation_generator, verbose = 1)


# In[56]:


preds = np.argmax(scores, axis = 1)


# In[57]:


preds 


# In[58]:


matrix = confusion_matrix(validation_generator.classes, preds)
disp = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = train_generator.class_indices)
disp = disp.plot(cmap = plt.cm.Blues)
plt.xticks(rotation = 90)
plt.show()


# In[60]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[0].set(xlim=(0, 80), ylim=(0, 2))
    ax[0].set_xlabel('Epochs')
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[1].set(xlim=(0, 80), ylim=(0, 1))
    ax[1].set_xlabel('Epochs')
    ax[0].legend()
    ax[1].legend()


# In[61]:


show_final_history(history)


# In[ ]:




