
from ast import Mod
import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use("Agg")
# Import packages
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import LocallyConnected1D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils as np_utils

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
# Implementation of LeNet-5 in keras 
# [LeCun et al., 1998. Gradient based learning applied to document recognition]
# Some minor changes are made to the architecture like using ReLU activation instead of 
# sigmoid/tanh, max pooling instead of avg pooling and softmax output layer 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="direccion del dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Nombre del Plot")
args = vars(ap.parse_args())

INIT_LR = 0
BS = 16
EPOCHS = 10

Hg = 64
Lng = 64

print("[INFO] Cargando imagenes...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (Hg, Lng))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


train_datagen = ImageDataGenerator(rotation_range=20, 
                            zoom_range=0.15, 
                            width_shift_range=0.2,  
                            height_shift_range=0.2,
                             shear_range=0.15, 
                             horizontal_flip=True,
                             vertical_flip = True,
                             brightness_range=None,
                            zca_whitening=False,
                            zca_epsilon=1e-06,
                             fill_mode="nearest")
l2_reg=0
weights=None

def get_alex_model():
    input_shape = Lng, Hg, 3
    input_ = Input(input_shape, name = 'the_input')

    conv_1 = Conv2D(16, kernel_size=(11, 11), padding='same', name = 'conv_1')(input_)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)
    mpool_1 = MaxPooling2D(pool_size = (2, 2), name = 'mpool_1')(relu_1)

    #Layer-2
    conv_2 = Conv2D(36, kernel_size = (5, 5), padding='same', name = 'conv_2')(mpool_1)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)
    mpool_2 = MaxPooling2D(pool_size = (2, 2), name = 'mpool_2')(relu_2)

    #Layer-3
    pad_1 = ZeroPadding2D((1, 1), name = 'pad_1')(mpool_2)
    conv_3 = Conv2D(64, (3, 3), padding='same', name = 'conv_3')(pad_1)
    norm_3 = BatchNormalization(name = 'norm_3')(conv_3)
    relu_3 = Activation('relu', name = 'relu_3')(norm_3)
    mpool_3 = MaxPooling2D(pool_size=(2, 2), name = 'mpool_3')(relu_3)

    #Layer-4
    pad_2 = ZeroPadding2D((1, 1), name = 'pad_2')(mpool_3)
    conv_4 = Conv2D(128, (3, 3), padding='same', name = 'conv_4')(pad_2)
    norm_4 = BatchNormalization(name = 'norm_4')(conv_4)
    relu_4 = Activation('relu', name = 'relu_4')(norm_4)

    #Layer-5
    pad_3 = ZeroPadding2D((1, 1), name = 'pad_3')(relu_4)
    conv_5 = Conv2D(128, (3, 3), padding='same', name = 'conv_5')(pad_3)
    norm_5 = BatchNormalization(name = 'norm_5')(conv_5)
    relu_5 = Activation('relu', name = 'relu_5')(norm_5)
    mpool_4 = MaxPooling2D(pool_size=(2, 2), name = 'mpool_4')(relu_5)

    #Layer-6
    flat_1 = Flatten(name = 'flat_1')(mpool_4)
    fc_1 = Dense(256, name = 'fc_1')(flat_1)
    norm_6 = BatchNormalization(name = 'norm_6')(fc_1)
    relu_6 = Activation('relu', name = 'relu_6')(norm_6)
    drop_1 = Dropout(0.5, name = 'drop_1')(relu_6)

    #Layer-7
    fc_2 = Dense(512, name ='fc_2')(drop_1)
    norm_7 = BatchNormalization(name = 'norm_7')(fc_2)
    relu_7 = Activation('relu', name = 'ralu_7')(norm_7)
    drop_2 = Dropout(0.5, name = 'drop_2')(relu_7)

    #Layer-8
    output = Dense(2, name = 'fc_3')(drop_2)
    model = Model(inputs = input_, outputs = output)
    model.summary()

    return model

def model_1():
    input_dim = (Lng, Hg, 3)
    input_ = Input(input_dim, name = 'the_input')
    output = Dense(2, name = 'output')(input_)
    model = Model(inputs = input_ , outputs = output)

    input_dim = np.expand_dims(input_dim, axis=0)
    model.summary()

    return model
    

def model_2():
    input_dim = (Lng, Hg, 3)
    input_ = Input(input_dim, name = 'the_input')
    layer1 = Dense(units=12, activation='sigmoid', name = 'layer_1')(input_)
    output = Dense(units = 2, activation = 'sigmoid')
    model = Model(inputs = input_ , outputs = output)
    input_dim = np.expand_dims(input_dim, axis=0)
    model.summary()

    return model

import numpy as np

def modelo_3():
    input_dim = (Lng, Hg, 3)
    input_ = Input(input_dim, name = 'the_input')
    layer1 = LocallyConnected2D(1, 2, strides= 2, activation= 'sigmoid', name = 'layer1')(input_)
    layer2 = LocallyConnected2D(1, 5, activation='sigmoid', name = 'layer2')(layer1)
    layer3 = Flatten(name='layer3')(layer2) 
    output = Dense(units=2, activation='sigmoid', name = 'output')(layer3)

    model = Model(inputs = input_, outputs = output)
    model.summary()
    input_dim = np.expand_dims(input_dim, axis=0)

    return model

def modelo_4():
    input_dim = (Lng, Hg, 3)
    input_ = Input(input_dim, name = 'the_input')
    layer1 = Conv2D(2, 2, strides= 2, activation= 'sigmoid', name = 'layer1')(input_)
    layer2 = LocallyConnected2D(1, 5, activation='sigmoid', name = 'layer2')(layer1)
    layer3 = Flatten(name='layer3')(layer2) 
    output = Dense(units=2, activation='sigmoid', name = 'output')(layer3)

    model = Model(inputs = input_, outputs = output)
    model.summary()
    input_dim = np.expand_dims(input_dim, axis=0)

    return model

def modelo_5():
    input_dim = (Lng, Hg, 3)
    input_ = Input(input_dim, name = 'the_input')
    layer1 = Conv2D(2, 2, strides= 2, activation= 'sigmoid', name = 'layer1')(input_)
    layer2 = Conv2D(4, 5, activation='sigmoid', name = 'layer2')(layer1)
    layer3 = Flatten(name='layer3')(layer2) 
    output = Dense(units=2, activation='sigmoid', name = 'output')(layer3)

    model = Model(inputs = input_, outputs = output)
    model.summary()
    input_dim = np.expand_dims(input_dim, axis=0)

    return model

model = modelo_4()




model.compile(loss="categorical_crossentropy", optimizer= "adam", metrics=["acc"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
# Train 
print("[INFO] Tr {} epochs...".format(EPOCHS))
H = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), callbacks=[reduce_lr], 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)
#callbacks=[reduce_lr],
# Evaluate 
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
model.save('m_0incan-wowave.h5')

# Plot 
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_accuracy")
plt.plot(N, H.history["val_acc"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
