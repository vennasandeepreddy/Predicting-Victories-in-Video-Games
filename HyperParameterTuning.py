import random
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

import optuna
from tensorflow.keras.constraints import MaxNorm
from optuna.visualization import plot_optimization_history


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

        # bring_batch_to_same_length(x_batch, max_length)
        # x_batch = np.array(x_batch, dtype='float32')
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


def objective(trial):
    length = (None, 8 * (2 + 3 + 12)) # maximun number of inputs
    neurons = trial.suggest_int('neurons', 10, 60)
    momentum = trial.suggest_float('momentum', 0.0, 1.0)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-3, log=True)
    initializers = trial.suggest_categorical('initializers',
                                             ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                                              'glorot_uniform', 'he_normal', 'he_uniform'])
    activation_methods = trial.suggest_categorical('activations',
                                                   ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
                                                    'hard_sigmoid', 'linear'])
    weight_constraints = trial.suggest_int('weight_constraints', 1, 5)
    EPOCHS = trial.suggest_int('epochs', 20, 30)
    BATCHSIZE = trial.suggest_int('batch', 64, 256)
    optimizers = trial.suggest_categorical('optimizer',
                                           ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])
    # n_layers = trial.suggest_int('n_layers', 1, 3)

    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=99.0, input_shape=length))

    for i in range(1):
        num_hidden = trial.suggest_int('n_units_l{}'.format(i), 4, length[1], log=True)

        model.add(LSTM(num_hidden, kernel_initializer=initializers, activation=activation_methods,
                        kernel_constraint=MaxNorm(weight_constraints)))

    model.add(Dense(1, activation='sigmoid'))

    model_list.append(model)
    model.compile(loss='binary_crossentropy', optimizer=optimizers, metrics=['accuracy'])

    history = model.fit_generator(generator(x_train, y_train, BATCHSIZE), epochs=EPOCHS,
                                  steps_per_epoch=len(x_train) // BATCHSIZE,
                                  validation_data=generator(x_val, y_val, BATCHSIZE),
                                  validation_steps=len(x_val) // BATCHSIZE)

    history_list.append(history)

    val_acc = np.array(history.history['val_accuracy'])

    return 1 - val_acc[-1]


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

model_list = []
history_list = []

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)

print(study.best_params)
print('')
print(study.best_value)
print('')
print(study.best_trial)

print(study.best_trial._number)
model_list[study.best_trial._number - 1].summary()

plot_optimization_history(study)

# Dropout
# Early stoping