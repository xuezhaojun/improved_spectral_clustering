from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

# the author of this file is ""

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

encoding_dim = 20
input_img = Input(shape=(784,))

encoded = Dense(50, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='softmax')(encoded)
decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(784, activation='linear')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded)

encoded_input = Input(shape=(encoding_dim,))
size = len(autoencoder.layers)
decoder_layer1 = autoencoder.layers[size-1]
decoder_layer2 = autoencoder.layers[size-2]

decoder = Model(inputs=encoded_input, outputs=decoder_layer1(decoder_layer2(encoded_input)))

def error(y_true, y_pred, z):
    return K.sum(K.square(y_pred-y_true),axis=-1)-K.sum(K.square(z),axis=-1)

def my_loss(z):
    def m_loss(y_true, y_pred):
        return error(y_true,y_pred,z)
    return m_loss;


model_loss = my_loss(encoded)
autoencoder.compile(optimizer='adadelta', loss=model_loss)

autoencoder.fit(x_train, x_train, epochs=5, batch_size=256,
                shuffle=True, validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

max_indexs = {}
for ps in encoded_imgs:
    max_index = 0
    index = 0
    max_value = ps[0]
    for p in ps:
        if p > max_value:
            max_value = p
            max_index = index
        index += 1
    max_indexs[max_index] = True
print(len(max_indexs))

plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test,s=3)
plt.colorbar()
plt.show()

n = 10000  # how many digits we will display
#plt.figure(figsize=(20, 4))
k = np.zeros([20])
c = 0
for i in range(n):
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)

#    ax = plt.subplot(2, n, i + 1 + n)
#    plt.imshow(decoded_imgs[i].reshape(28, 28))
#    ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)

    k[np.argmax(encoded_imgs[i])] = k[np.argmax(encoded_imgs[i])] + 1
    c = c + np.argmax(encoded_imgs[i]);
#plt.show()
print(c/10000)
for i in range(20):
    if (k[i]>500):
        print(k[i])
