import keras
import keras.layers as layers
from keras.datasets import mnist
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Data initialization
(train_data, train_labels), (test_data, test_labels) = mnist.load_data('data')

train_data = train_data.reshape((60000, 28 * 28)).astype('float32') / 255
test_data = test_data.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Learning model
network = keras.models.Sequential()
network.add(layers.Dense(28 * 28, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# Argument initialization and training
optimizer = optimizers.rmsprop(lr=0.001)
loss = losses.categorical_crossentropy
metrics = ['accuracy']

network.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = network.fit(train_data, train_labels, epochs=20, batch_size=128)

# Testing
_, accuracy = network.evaluate(test_data, test_labels)

print('\n', round(accuracy * 100, 2), '%')
