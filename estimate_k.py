from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import tensorflow as tf
import gmeans
tf.config.run_functions_eagerly(True)

# unpper_bound : k_u in paper
def estimate(upper_bound,train):
    input = Input(shape=(train.shape[1],))
    encoded = Dense(50, activation='relu')(input)
    encoded = Dense(upper_bound, activation='softmax')(encoded)
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(train.shape[1], activation='linear')(decoded)

    autoencoder = Model(input,decoded)
    encoder = Model(input,encoded)

    autoencoder.compile(optimizer='sgd',loss=clusters.loss_func_in_scde_plus(encoded))
    autoencoder.fit(train,train,epochs=50, batch_size=256, shuffle=True)

    return get_k_from_result(encoder.predict(train))

# not used now
# unpper_bound : k_u in paper
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
        if max_indexs.get(max_index) == None:
            max_indexs[max_index] = 1
        else:
            max_indexs[max_index] += 1
    return len(max_indexs)

def SS(begin, end, dataset):
    k = begin
    max_score = 0
    max_k = -1
    while k <= end:
        km = KMeans(n_clusters=k, max_iter=200)
        km_result = km.fit_predict(dataset)
        score = silhouette_score(dataset,km_result)
        if score > max_score:
            max_k = k
            max_score = score
        k+=1
    return max_k

def DB(begin, end, dataset):
    k = begin
    max_score = 0
    max_k = -1
    while k <= end:
        km = KMeans(n_clusters=k, max_iter=200)
        km_result = km.fit_predict(dataset)
        score = davies_bouldin_score(dataset,km_result)
        if score > max_score:
            max_k = k
            max_score = score
        k+=1
    return max_k

def estimate_gmeans(dataset):
    g = gmeans.Gmeans(min_obs=500, random_state=1010, max_depth=1000,strictness=4)
    g.fit(dataset)
    return len(set(list(g.labels_)))