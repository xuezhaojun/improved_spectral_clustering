import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.cluster import SpectralClustering

# standard squered error is a loss function
def sse(y_true, y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1) # TODO what is -1?

# z: the output of inner layer
def loss_func_in_scde_plus(z):
    def loss(y_true, y_pred):
        return sse(y_true,y_pred) - K.sum(K.square(z),axis=-1)
    return loss

# dimenstion reduction using autoencoder
def clustering(train,k):
    # make autoencoder
    encoding_dim = 32
    input = keras.Input(shape=(784,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input)
    decoded = layers.Dense(784, activation="sigmoid")(encoded) 
    
    # creater autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)

    # train data
    autoencoder.compile(loss=sse, optimizer='sgd')
    autoencoder.fit(train, train, epochs=50, batch_size=256,shuffle=True)

    # use encoder to encode raw data
    encoded_data = encoder.predict(train)
   
    # clustering
    print("begin sc clustering")
    sc = SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors")
    sc_result = sc.fit_predict(encoded_data[:5000]) # TODO 跑60000个数据还是不行，时间超久
    print("end sc clustering")
    return sc_result