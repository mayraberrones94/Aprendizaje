
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
EPOCHS = 20

Hg = 224
Lng = 224

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

# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img  = batch_img[:,0::2]
    even_img = batch_img[:,1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H

def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H

def Wavelet(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:,0]
    g = batch_image[:,1]
    b = batch_image[:,2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)


    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
                    g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                    b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

  
    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
 
    return [decom_level_1, decom_level_2]


def Wavelet_out_shape(input_shapes):
    # print('in to shape')
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12])]

def get_wavelet_cnn_model():
  input_shape = 224, 224, 3
  input_ = Input(input_shape, name='the_input')
  wavelet = Lambda(Wavelet, Wavelet_out_shape, name='wavelet')
  input_l1, input_l2 = wavelet(input_)

  conv_1 = Conv2D(16, kernel_size=(11, 11), padding='same', name = 'conv_1')(input_l1)
  norm_1 = BatchNormalization(name='norm_1')(conv_1)
  relu_1 = Activation('relu', name='relu_1')(norm_1)

  conv_1_2 = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='same', name='conv_1_2')(relu_1)
  norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
  relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

  conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_a')(input_l2)
  norm_a = BatchNormalization(name='norm_a')(conv_a)
  relu_a = Activation('relu', name='relu_a')(norm_a)

  # concate level one and level two decomposition
  concate_level_2 = concatenate([relu_1_2, relu_a])
  conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_2')(concate_level_2)
  norm_2 = BatchNormalization(name='norm_2')(conv_2)
  relu_2 = Activation('relu', name='relu_2')(norm_2)       

  conv_2_2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_2_2')(relu_2)
  norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
  relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

  pool_5_1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', name='max_pool_5_1')(relu_2_2)
  flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 

  fc_5 = Dense(256, name='fc_5')(flat_5_1)
  norm_5 = BatchNormalization(name='norm_5')(fc_5)
  relu_5 = Activation('relu', name='relu_5')(norm_5)
  drop_5 = Dropout(0.5, name='drop_5')(relu_5)

  fc_6 = Dense(512, name='fc_6')(drop_5)
  norm_6 = BatchNormalization(name='norm_6')(fc_6)
  relu_6 = Activation('relu', name='relu_6')(norm_6)
  drop_6 = Dropout(0.5, name='drop_6')(relu_6)

  output = Dense(2, activation='softmax', name='fc_7')(drop_6)

  model = Model(inputs=input_, outputs=output)
  model.summary()
  #plot_model(model, to_file='wavelet_cnn_0.5.png')

  return model

model = get_wavelet_cnn_model()



model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["acc"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
# Train 
print("[INFO] Tr {} epochs...".format(EPOCHS))
H = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=BS), 
                                    validation_data=(testX, testY), 
                                    callbacks=[reduce_lr], 
                                    validation_steps = 1000,
                                    steps_per_epoch=1000, 
                                    epochs=EPOCHS)
#callbacks=[reduce_lr],
# Evaluate 
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
model.save('m_0incanwave.h5')

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