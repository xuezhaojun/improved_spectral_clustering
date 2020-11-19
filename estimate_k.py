from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import clusters
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# unpper_bound : k_u in paper
def estimate_mnist(upper_bound,train,test):
    input = Input(shape=(train.shape[1],))
    encoded = Dense(50, activation='relu')(input)
    encoded = Dense(upper_bound, activation='softmax')(encoded)
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(train.shape[1], activation='linear')(decoded)

    autoencoder = Model(input,decoded)
    encoder = Model(input,encoded)

    autoencoder.compile(optimizer='sgd',loss=clusters.loss_func_in_scde_plus(encoded))
    # 内存原因：50个epochs会报错
    autoencoder.fit(train,train,epochs=40, batch_size=256, shuffle=True,validation_data=(train,train))

    return encoder.predict(test)

# unpper_bound : k_u in paper
def get_k_from_result(unpper_bound,result):
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
        if max_indexs.get(max_index) == None:
            max_indexs[max_index] = 1
        else:
            max_indexs[max_index] += 1
    count = 0
    for _,v in max_indexs.items():
        if len(v) > (1/unpper_bound)*len(result):
            count+=1
    return count