from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import scde
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

# unpper_bound : k_u in paper
def estimate_mnist(upper_bound,train):
    input = Input(shape=(784,))
    encoded = Dense(50, activation='relu')(input)
    encoded = Dense(upper_bound, activation='softmax')(encoded)
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(784, activation='linear')(decoded)

    autoencoder = Model(input,decoded)
    encoder = Model(input,encoded)

    autoencoder.compile(optimizer='sgd',loss=scde.loss_func_in_scde_plus(encoded))
    # epochs = 50 一般,为了计算快速，目前设置为5
    autoencoder.fit(train,train,epochs=5, batch_size=256, shuffle=True,validation_data=(train,train))

    return encoder.predict(train)

def get_k_from_result(result):
    # count for k
    max_indexs = {}
    for ps in result:
        max_index = 0
        index = 0
        max_value = ps[0]
        for p in ps:
            if p > max_value:
                max_value = p
                max_index = index
            index += 1
        max_indexs[max_index] = True
    return len(max_indexs)