from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import random
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# Model configuration
features = 8 * (2 + 3 + 12) # features of 8 units
batch_size = 128
loss_function = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
units = 64
epochs = 10

def generator(xData, yData, batch_size):
    samples_per_epoch = len(xData)
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while True:
        x_batch = xData[batch_size * counter:batch_size * (counter + 1)]
        y_batch = np.array(yData[batch_size * counter:batch_size * (counter + 1)], dtype='float32')
        x_batch = tf.keras.preprocessing.sequence.pad_sequences(x_batch, padding="post", value=99.0, dtype='float32')
        counter += 1
        yield x_batch, y_batch

        if counter >= number_of_batches:
            data = list(zip(xData, yData))
            random.shuffle(data)
            xData, yData = zip(*data)
            xData = list(xData)
            yData = list(yData)
            counter = 0

x = []
y = []

with open('x.pkl', 'rb') as f:
    x = pickle.load(f)

with open('y.pkl', 'rb') as f:
    y = pickle.load(f)

data = list(zip(x, y))
random.shuffle(data)
x, y = zip(*data)

x_train = list(x[:(4 * len(x))//5])
x_val = list(x[(4*len(x))//5:])
y_train = list(y[:(4 * len(x))//5])
y_val = list(y[(4*len(x))//5:])

model = Sequential()
model.add(tf.keras.layers.Masking(mask_value=99.0, input_shape=(None, features)))
model.add(tf.keras.layers.GRU(units, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=loss_function,
              optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC()])

history = model.fit_generator(generator(x_train, y_train, batch_size), epochs=epochs,
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
