import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt

# standard squered error is a loss function
def sse(y_true, y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1)

# z: the output of inner layer
def loss_func_in_scde_plus(z):
    def loss(y_true, y_pred):
        return sse(y_true,y_pred) - K.sum(K.square(z),axis=-1)
    return loss

def train_autoencoder(train,innermost):
    # make autoencoder
    input_dim = train.shape[1]
    
    input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(500, activation="relu")(input)
    encoded = layers.Dense(500, activation='relu')(encoded)
    encoded = layers.Dense(2000, activation='relu')(encoded)
    encoded = layers.Dense(innermost, activation='sigmoid')(encoded)

    decoded = layers.Dense(2000, activation='relu')(encoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation="relu")(decoded) 

    # creater autoencoder
    autoencoder = keras.Model(input, decoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='sgd')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    return autoencoder

def show_gray_img(img,embeded_img):
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(embeded_img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def de(train,test):
    # make autoencoder
    input_dim = train.shape[1]
    
    input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(500, activation="relu")(input)
    encoded = layers.Dense(500, activation='relu')(encoded)
    encoded = layers.Dense(2000, activation='relu')(encoded)
    encoded = layers.Dense(10, activation='sigmoid')(encoded)

    decoded = layers.Dense(2000, activation='relu')(encoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation="relu")(decoded) 

    # creater autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='sgd')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    return encoder.predict(test)

def de_smaller(train,test):
    # make autoencoder
    input_dim = train.shape[1]
    
    input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(200, activation="relu")(input)
    encoded = layers.Dense(200, activation='relu')(encoded)
    encoded = layers.Dense(800, activation='relu')(encoded)
    encoded = layers.Dense(10, activation='sigmoid')(encoded)

    decoded = layers.Dense(800, activation='relu')(encoded)
    decoded = layers.Dense(200, activation='relu')(decoded)
    decoded = layers.Dense(200, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation="relu")(decoded) 

    # creater autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='sgd')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    return encoder.predict(test)

def de_with_adam(train,test):
    # make autoencoder
    input_dim = train.shape[1]
    
    input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(500, activation="relu")(input)
    encoded = layers.Dense(500, activation='relu')(encoded)
    encoded = layers.Dense(2000, activation='relu')(encoded)
    encoded = layers.Dense(10, activation='sigmoid')(encoded)

    decoded = layers.Dense(2000, activation='relu')(encoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation="relu")(decoded) 

    # creater autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='adam')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    return encoder.predict(test)

def score(sc_result, lable_true):
    # get nmi
    nmi = normalized_mutual_info_score(lable_true,sc_result)
    # get ari
    ari = adjusted_rand_score(lable_true, sc_result)
    return nmi, ari