import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout
import matplotlib.pyplot as plt


def getData():
    (train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_X.shape))
    print('Y_test:  ' + str(test_y.shape))


    shape_train = train_X[0].shape
    input_shape = (shape_train[0], shape_train[1], 1)
    shape_test = test_X[0].shape
    input_shape_test = (shape_test[0], shape_test[1], 1)
    x_train = np.array(train_X).reshape(len(train_X), input_shape[0], input_shape[1], input_shape[2])
    x_test = np.array(test_X).reshape(len(test_X), input_shape_test[0], input_shape_test[1], input_shape_test[2])

    return (x_train, train_y), (x_test, test_y), input_shape

def plot_loss(loss_train, loss_val, epochs):
    plt.plot(epochs, loss_train, 'g', label='Zbiór treningowy')
    plt.plot(epochs, loss_val, 'b', label='Zbiór walidacyjny')
    plt.title('Średnia strata')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.ylim([0, 2.5])
    plt.show()


def plot_acc(acc_train, acc_val, epochs):
    plt.plot(epochs, acc_train, 'g', label='Zbiór treningowy')
    plt.plot(epochs, acc_val, 'b', label='Zbiór walidacyjny')
    plt.title('Średnia poprawność')
    plt.xlabel('Epoka')
    plt.ylabel('Poprawność')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()
# tf.keras.layers.Conv2D(
#     filters,
#     kernel_size -
#     strides=(1, 1),
#     padding="valid" -  "valid" means no padding.
#     data_format=None,
#     dilation_rate=(1, 1),
#     groups=1,
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )


def fc(shape):
    model = tf.keras.models.Sequential([
        Flatten(input_shape=shape),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


def cnn(shape):
#    model = tf.keras.models.Sequential([
 #       Conv2D(5, kernel_size=4, strides=(2, 2), activation='relu', input_shape=shape),
        #MaxPooling2D(pool_size=2),
  #      Flatten(),
   #     Dense(10, activation='softmax')
    #])

    model = tf.keras.models.Sequential([
        Conv2D(15, kernel_size=3, strides=(1, 1), activation='relu', input_shape=shape),
        MaxPooling2D(pool_size=2),
        Dropout(0.1),
        Conv2D(15, kernel_size=3, strides=(1, 1), activation='relu', input_shape=shape),
        MaxPooling2D(pool_size=2),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


def run():
    epochs = 10
    train_loop = 10
    loss_train_sum_cnn = np.zeros(train_loop)
    loss_test_sum_cnn = np.zeros(train_loop)
    acc_train_sum_cnn = np.zeros(train_loop)
    acc_test_sum_cnn = np.zeros(train_loop)
    loss_train_sum_fc =np.zeros(train_loop)
    loss_test_sum_fc = np.zeros(train_loop)
    acc_train_sum_fc = np.zeros(train_loop)
    acc_test_sum_fc = np.zeros(train_loop)

    for i in range(train_loop):
        (x_train, y_train), (x_test, y_test), input_shape = getData()

        model_cnn = cnn(input_shape)
        history_cnn = model_cnn.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

#        model_fc = fc(input_shape)
#        history_fc = model_fc.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

        loss_train_sum_cnn = np.add(loss_train_sum_cnn, np.array(history_cnn.history['loss']))
        loss_test_sum_cnn = np.add(loss_test_sum_cnn, np.array(history_cnn.history['val_loss']))
        acc_train_sum_cnn = np.add(acc_train_sum_cnn, np.array(history_cnn.history['accuracy']))
        acc_test_sum_cnn = np.add(acc_test_sum_cnn, np.array(history_cnn.history['val_accuracy']))

#        loss_train_sum_fc = np.add(loss_train_sum_fc, np.array(history_fc.history['loss']))
#        loss_test_sum_fc = np.add(loss_test_sum_fc, np.array(history_fc.history['val_loss']))
#        acc_train_sum_fc = np.add(acc_train_sum_fc, np.array(history_fc.history['accuracy']))
#        acc_test_sum_fc = np.add(acc_test_sum_fc, np.array(history_fc.history['val_accuracy']))

    epochs = range(1, epochs+1)
    print(loss_train_sum_cnn/train_loop)
    print(loss_test_sum_cnn/train_loop)
    print(loss_train_sum_fc/train_loop)
    print(loss_test_sum_fc/train_loop)
    print(acc_train_sum_cnn/train_loop)
    print(acc_test_sum_cnn/train_loop)
    print(acc_train_sum_fc/train_loop)
    print(acc_test_sum_fc/train_loop)

    plot_loss(loss_train_sum_cnn/train_loop, loss_test_sum_cnn/train_loop, epochs)
#    plot_loss(loss_train_sum_fc/train_loop, loss_test_sum_fc/train_loop, epochs)
    plot_acc(acc_train_sum_cnn/train_loop, acc_test_sum_cnn/train_loop, epochs)
 #   plot_acc(acc_train_sum_fc/train_loop, acc_test_sum_fc/train_loop, epochs)



if __name__ == '__main__':
    run()
