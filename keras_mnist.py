import keras
import keras.layers as layers
from keras.datasets import mnist
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os


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
history = network.fit(train_data, train_labels, epochs=5, batch_size=128)

# Testing
_, accuracy = network.evaluate(test_data, test_labels)

clear = lambda: os.system('clear')

while True:
    clear()
    print('\n', round(accuracy * 100, 2), '%')
    print('There are 10000 handwritten numbers from MNIST dataset.')
    print('Neural network cat predict any of this numbers.')

    try:
        index = int(input('Type any index: '))
    except:
        index = int(input('Type any INTEGER index: '))

    print('Neural network thinks its', end=' ')
    print(np.argmax(network.predict(test_data)[index]))

    plt.imshow(test_data[index].reshape((28, 28)))
    plt.show()
    input('Press Enter to continue: ')