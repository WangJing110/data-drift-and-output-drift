#Load libraries
import keras.metrics
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.metrics import mse, mae, mape
from keras import backend as K, metrics
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from case_1 import PSI
import warnings
warnings.filterwarnings("ignore")


#Specify the architecture for the auto encoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn import logger

"""
    PATHS
"""
train_path = "./dataset/china_dark/train/"
test_path = "./dataset/china_dark/test/"
gm_test_path = "./dataset/german/test/"

input_img = Input(shape=(96, 96, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss="mean_squared_error")
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# (x_train, _), (x_test, _) = mnist.load_data()
datagen = ImageDataGenerator()
# x_train = datagen.flow_from_directory(train_path,
#                                       target_size=(96, 96), color_mode="rgb", batch_size=5000, seed=0)
# x_train, _ = next(x_train)
x_test = datagen.flow_from_directory(test_path,
                                     target_size=(96, 96), color_mode="rgb", batch_size=5000, seed=0)
x_test, _ = next(x_test)

# x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 96, 96, 3))
x_test = np.reshape(x_test, (len(x_test), 96, 96, 3))

# Generate some noisy data
noise_factor = 0.2
# x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = 0.8*x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# K.set_value(autoencoder.optimizer.lr, 0.1)
# keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0)
# history = autoencoder.fit(x_train_noisy, x_train,
#                           epochs=20,
#                           batch_size=128,
#                           shuffle=True,
#                           validation_split=0.2).history
# print(history['val_loss'][-1])

# autoencoder.save("./noise_autoencoder.pth")
autoencoder = keras.models.load_model("./noise_autoencoder.pth")

decoded_imgs = autoencoder.predict(x_test_noisy)
x_test_noisy_loss = autoencoder.evaluate(x_test_noisy, x_test)
print(x_test_noisy_loss)

# decoded_imgs = autoencoder.predict(x_train)
# x_train_loss = autoencoder.evaluate(x_train, decoded_imgs)
# print(x_train_loss)
#
# decoded_imgs = autoencoder.predict(x_test)
# x_test_loss = autoencoder.evaluate(x_test, decoded_imgs)
# print(x_test_loss)

other_data = datagen.flow_from_directory(gm_test_path, target_size=(96, 96), color_mode="rgb", batch_size=5000, seed=5
                                         )
other_data, _ = next(other_data)
other_data = other_data.astype('float32') / 255.
other_data = np.reshape(other_data, (len(other_data), 96, 96, 3))
other_data_noisy = 0.8*other_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=other_data.shape)
other_data_pred = autoencoder.predict(other_data)
other_data_noisy = np.clip(other_data_noisy, 0., 1.)
other_data_loss = autoencoder.evaluate(other_data_noisy, other_data)
print(other_data_loss)

#Take a look at the reconstructed images
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    # ax = plt.subplot(3, n, i)
    # plt.imshow(x_test_noisy[i].reshape(96, 96, 3))
    # # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display original
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test[i].reshape(96, 96, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n*2)
    plt.imshow(decoded_imgs[i].reshape(96, 96, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle("China loss: " + str(x_test_noisy_loss), fontsize=14, fontweight='bold')
plt.show()

#Take a look at the reconstructed images
decoded_imgs = autoencoder.predict(other_data)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    # ax = plt.subplot(3, n, i)
    # plt.imshow(other_data_noisy[i].reshape(96, 96, 3))
    # # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display original
    ax = plt.subplot(3, n, i + n)
    plt.imshow(other_data[i].reshape(96, 96, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n*2)
    plt.imshow(decoded_imgs[i].reshape(96, 96, 3))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle("German loss: " + str(other_data_loss), fontsize=14, fontweight='bold')
plt.show()

#ouput 
psi = PSI()
d1 = np.array([0.19633333, 0.07583333, 0.16933333, 0.25483333, 0.30366667])
d2 = np.array([0.16173333, 0.2356, 0.2116, 0.1884, 0.20266667])
ouput_psi = psi.calculate_psi(d1, d2, 10, isrange=True)
