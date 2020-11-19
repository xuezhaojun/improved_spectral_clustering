import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# standard squered error is a loss function
def sse(y_true, y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1) # TODO what is -1?

# z: the output of inner layer
def loss_func_in_scde_plus(z):
    def loss(y_true, y_pred):
        return sse(y_true,y_pred) - K.sum(K.square(z),axis=-1)
    return loss

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
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded) 
    
    # creater autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='sgd')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    return encoder.predict(test)

def score(sc_result, lable_true):
    # get nmi
    nmi = normalized_mutual_info_score(lable_true,sc_result)
    # get ari
    ari = adjusted_rand_score(lable_true, sc_result)
    return nmi, ari