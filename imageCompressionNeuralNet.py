from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

DATA = "./images/"

def save_mnist():
    (x_train, _), (x_test, _) = mnist.load_data()

    i = 0
    for img in x_test:
        im = Image.fromarray(img)
        im.save(DATA + "img" + str(i) + ".jpeg")
        i += 1

def load_mnist():
    x_test = []
    for file in os.listdir(DATA):
        if file.endswith(".jpeg"):
            img = Image.open(DATA + file)
            x_test.append(np.array(img.copy()))
            img.close()
    return x_test


input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_test = np.array(load_mnist())

print(x_test.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

def train(autoencoder):
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    autoencoder_json = autoencoder.to_json()
    with open("autoencoder.json", "w") as json_file:
        json_file.write(autoencoder_json)
    autoencoder.save_weights("model.h5")
    print("Saved autoencoder model to disk")

# train(autoencoder)

json_file = open('autoencoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

decoded_imgs = loaded_model.predict(x_test)
print("Loaded autoencoder model from disk")

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    f, (ax1, ax2) = plt.subplots(1, 2)
    #plt.imshow(x_test[i].reshape(28, 28))
    #plt.gray()
    ax1.imshow(x_test[i].reshape(28, 28))
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    #ax = plt.subplot(2, n, i + n)
    ax2.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
plt.show()
