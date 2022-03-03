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


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax1.set_title('Model AUC')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax1.legend(['Train', 'Validation'], loc='upper left')

fig.tight_layout

plt.show()

steps_per_epoch = np.math.ceil(val_iterator.samples / val_iterator.batch_size)
predictions = model.predict(val_iterator, steps=steps_per_epoch)

y_true = val_iterator.classes
y_pred = np.argmax(predictions, axis=1)

class_labels = list(val_iterator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true,y_pred, target_names=class_labels)

print("Confusion Matrix: ", cm)
print("Classifcation Metrics Report: ", report)