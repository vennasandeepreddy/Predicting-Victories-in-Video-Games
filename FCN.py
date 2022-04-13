import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# Model configuration
features = 8 * (2 + 3 + 12)
batch_size = 128
loss_function = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
number_epochs = 10

def bring_batch_to_same_length(x_batch, max_length):
    for game in x_batch:
        for i in range(len(game), max_length):
            game.append(game[-1])


def generator(xData, yData, batch_size):
    samples_per_epoch = len(xData)
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while True:
        x_batch = xData[batch_size * counter:batch_size * (counter + 1)]
        y_batch = np.array(yData[batch_size * counter:batch_size * (counter + 1)], dtype='float32')

        bring_batch_to_same_length(x_batch, max_length)
        x_batch = np.array(x_batch, dtype='float32')
        # x_batch = tf.keras.preprocessing.sequence.pad_sequences(x_batch, padding="post", value=99.0, dtype='float32')

        counter += 1
        yield x_batch, y_batch

        if counter >= number_of_batches:
            data = list(zip(xData, yData))
            random.shuffle(data)
            xData, yData = zip(*data)
            xData = list(xData)
            yData = list(yData)
            counter = 0

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load dataset
x = []
y = []

with open('x.pkl', 'rb') as f:
    x = pickle.load(f)

with open('y.pkl', 'rb') as f:
    y = pickle.load(f)

max_length = 0
for game in x:
    max_length = max(max_length, len(game))

data = list(zip(x, y))
random.shuffle(data)
x, y = zip(*data)

x_train = list(x[:(4 * len(x))//5])
x_val = list(x[(4*len(x))//5:])
y_train = list(y[:(4 * len(x))//5])
y_val = list(y[(4*len(x))//5:])

input_shape = (max_length, features)

input_layer = keras.layers.Input(input_shape)

# Reference for the model: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
# From the following paper: H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller,
# "Deep learning for time series classification: a review," Data Mining and Knowledge Discovery,
# vol. 33, no. 4, Springer US, 2019, pp. 917–63, doi: 10.1007/s10618-019-00619-1.

# ********* Slightly modified to accomodate for binary classification *********

conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation(activation='relu')(conv1)

conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

# FINAL

gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy',tf.keras.metrics.AUC()])

# Give the model summary
model.summary()

# Train the model
history = model.fit_generator(generator(x_train, y_train, batch_size), epochs=number_epochs,
                              steps_per_epoch=len(x_train)//batch_size,
                              validation_data=generator(x_val, y_val, batch_size),
                              validation_steps=len(x_val)//batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["training_acc", "validation_acc"])
plt.show()

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.legend(["training_AUC", "validation_AUC"])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["training_loss", "validation_loss"])
plt.show()
