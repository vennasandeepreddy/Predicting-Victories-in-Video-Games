import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

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
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

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

# Reference for the model: https://github.com/titu1994/MLSTM-FCN/blob/master/acvitivity_model.py

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

xx = Permute((2, 1))(input_layer)
xx = Masking()(xx)
xx = LSTM(8)(xx)
xx = Dropout(0.8)(xx)

yy = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(input_layer)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)
yy = squeeze_excite_block(yy)

yy = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(yy)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)
yy = squeeze_excite_block(yy)

yy = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(yy)
yy = BatchNormalization()(yy)
yy = Activation('relu')(yy)

yy = GlobalAveragePooling1D()(yy)

xx = concatenate([xx, yy])

out = Dense(1, activation='sigmoid')(xx)

model = Model(input_layer, out)
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
