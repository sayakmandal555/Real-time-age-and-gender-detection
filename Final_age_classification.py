#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical


# In[2]:


os.chdir('F:\Age_gender\Face')


# In[3]:


im =Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128,128))
im


# In[4]:


onlyfiles = os.listdir()


# In[5]:


len(onlyfiles)


# In[6]:


shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]


# In[7]:


classes = []
for i in age:
    i = int(i)
    if i <= 14:
        classes.append(0)
    if (i>14) and (i<=25):
        classes.append(1)
    if (i>25) and (i<40):
        classes.append(2)
    if (i>=40) and (i<60):
        classes.append(3)
    if i>=60:
        classes.append(4)
    


# In[8]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face =cv2.resize(face, (32, 32) )
    X_data.append(face)


# In[9]:


X = np.squeeze(X_data)


# In[10]:


X.shape


# In[11]:


X = X.astype('float32')
X /= 255


# In[12]:


classes[:10]


# In[13]:


categorical_labels = to_categorical(classes, num_classes=5)


# In[14]:


categorical_labels[:10]


# In[15]:


(x_train, y_train), (x_test, y_test) = (X[:15008],categorical_labels[:15008]) , (X[15008:] , categorical_labels[15008:])
(x_valid , y_valid) = (x_test[:7000], y_test[:7000])
(x_test, y_test) = (x_test[7000:], y_test[7000:])


# In[16]:


len(x_train)+len(x_test) + len(x_valid) == len(X)


# In[17]:



model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(5, activation='softmax'))


# Take a look at the model summary
model.summary()


# In[18]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[19]:


model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=20,
         validation_data=(x_valid, y_valid),)


# In[20]:


score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


# In[21]:



labels =["CHILD",  # index 0
        "YOUTH",      # index 1
        "ADULT",     # index 2 
        "MIDDLEAGE",        # index 3 
        "OLD",         # index 4
        ]


# In[22]:


y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()


# In[23]:


model.save('F:\Age_gender\Age_Classification_model.h5')


# In[ ]:




