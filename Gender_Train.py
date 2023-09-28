import tensorflow as tf #deep learning models,neural network
import keras #serve interface
from keras.preprocessing.image import ImageDataGenerator #preprocess imagedata during model training
from keras.optimizers import Adam #training deep learning models.
from tensorflow.keras.utils import img_to_array
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import BatchNormalization  #deep neural networks to normalize the intermediate activations of a model.
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation #introduce nonlinearity
from keras.layers import Flatten 
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob #specific file pattern

# initial parameters
epochs = 100
learning_rate = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'F:\Minor_project\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

#convert images to arrays and labelling 
for img in image_files:
    image=cv2.imread(img)

    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] #F:\Gender\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append([label]) # [[1], [0], [0], ...]

# pre-processing
data = np.array(data, dtype="float") / 255.0   #image pixel value is 1 to 255 . so we divide by 255 then pixel is 0 to 1.
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY =to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY =to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,     
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
#The ImageDataGenerator class allows your model to receive new variations of the images at each epoch. 
#IMAGE data generator change the properties of images

# define model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1  #chandim is used for normaliztion

    if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first", 
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape)) #32 filters,image size 3,3
    model.add(Activation("relu"))   #relu is non linear function
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))  #maxpolling2D is remove noises
    model.add(Dropout(0.25))  #dropout avoid overfitting

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) #flatten is used for dence layer. convert multidimentional to 1d
    model.add(Dense(1024)) #1024 neurons
    model.add(Activation("relu"))#relu activation function,for image classification
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  #50% neurons are deactivated during front and back propagation for avoiding overfitting

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))    #output layer

    return model
# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
opt = Adam(learning_rate=learning_rate, decay=learning_rate/epochs) #adam optimizer
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])  #binary_crossentropy loss function

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)
# save the model to disk
model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')
