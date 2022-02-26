from gc import callbacks
from msilib.schema import Directory
from pandas import Categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np 

BS = 32
directory = "Covid19-dataset/train"

train_data_gen = ImageDataGenerator( 
    rescale=1./255,
    zoom_range=0.1,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
)

train_iterator = train_data_gen.flow_from_directory(
    "Covid19-dataset/train",
    class_mode='categorical', 
    color_mode='grayscale',
    target_size=(256,256),
    batch_size=BS
)

train_iterator.next()

val_data_gen = ImageDataGenerator(rescale=1./255)

val_iterator = val_data_gen.flow_from_directory(
    "Covid19-dataset/train", 
    class_mode='categorical', 
    color_mode='grayscale'
)
def design_model(input):
    model = Sequential()
    model.add(tf.keras.Input(shape=(256,256,1)))

    model.add(layers.Conv2D(8, 3, strides=2, activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(8, 3, strides=1, activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(layers.Dropout(0.1))
    # model.add(layers.Conv2D(8, 3, strides=2, activation='relu'))
    # model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(layers.Flatten())

    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()
    return model

model = design_model(train_iterator)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

es = EarlyStopping(monitor='val_auc', patience=30, verbose=1, mode='min')

history = model.fit(
    train_iterator,
    steps_per_epoch = train_iterator.samples / BS,
    epochs = 45,
    validation_data = val_iterator,
    validation_steps = val_iterator.samples / BS,
    callbacks=[es]
)

