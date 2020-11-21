import numpy as np
from tensorflow.python.keras.activations import swish
from tensorflow.python.keras.backend import switch
import clusters
import datasets as ds
import estimate_k as ek

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from pymongo import MongoClient

# prepare datasets

# uci
uci_data, uci_target = ds.get_uci()
uci_data = uci_data.astype('float32') / 16.

# usps
usps_data, _, usps_test, usps_test_lable = ds.get_usps()

# mnist
mnist_train,_,mnist_test,mnist_test_lable = ds.get_mnist()
mnist_train = mnist_train.astype('float32') / 255.
mnist_train = mnist_train.reshape((len(mnist_train),np.prod(mnist_train.shape[1:]))) # make 28*28 img to 784 demission matrix
mnist_test = mnist_test.astype('float32') / 255.
mnist_test = mnist_test.reshape((len(mnist_test), np.prod(mnist_test.shape[1:])))

# fashion
fashion_train,_,fashion_test,fashion_test_lable = ds.get_fashion_mnist()
fashion_train = fashion_train.astype('float32') / 255.
fashion_train = fashion_train.reshape((len(fashion_train),np.prod(fashion_train.shape[1:]))) # make 28*28 img to 784 demission matrix
fashion_test = fashion_test.astype('float32') / 255.
fashion_test = fashion_test.reshape((len(fashion_test), np.prod(fashion_test.shape[1:])))

# cifar10
cifar_train, _, cifar_test, cifar_test_lable = ds.get_cifar10()
cifar_train = cifar_train.astype('float32') / 255.
cifar_train = cifar_train.reshape((len(cifar_train),32*32*3))
cifar_test = cifar_test.astype('float32') / 255.
cifar_test = cifar_test.reshape((len(cifar_test),32*32*3))
cifar_test_lable_1d = []
for lable in cifar_test_lable:
    cifar_test_lable_1d.append(lable[0])

def process(dataset):
    # get train,test,test_talbe,true_k by dataset
    train, test, test_lable, true_k = init_dataset(dataset)

    estimate = {}
    cluster_result = {}

    # DE
    de_result = clusters.de(train,test)

    # --- estimate k ---

    # DE+SA
    k = ek.estimate(2*true_k, de_result)
    estimate["DE+SA"] = k

    # SA
    k = ek.estimate(2*true_k,test)
    estimate["SA"] = k

    # --- clustering ---

    # K-means
    km = KMeans(n_clusters=true_k, max_iter=200)
    km_result = km.fit_predict(test)
    score = clusters.score(test_lable,km_result)
    cluster_result["K-means"] = score

    # DE, K-means
    km = KMeans(n_clusters=true_k, max_iter=200)
    de_km_result = km.fit_predict(de_result)
    score = clusters.score(test_lable,de_km_result)
    cluster_result["DE+K-means"] = score

    # SC
    sc = SpectralClustering(n_clusters=true_k, eigen_solver='arpack', affinity="nearest_neighbors") 
    sc_result = sc.fit_predict(test)
    score = clusters.score(test_lable,sc_result)
    cluster_result["SC"] = score

    # DE, SC
    sc = SpectralClustering(n_clusters=true_k, eigen_solver='arpack', affinity="nearest_neighbors")
    scde_result = sc.fit_predict(de_result)
    score = clusters.score(test_lable,scde_result)
    cluster_result["SCDE"] = score

    result = {}
    result["estimate"] = estimate
    result["cluster_result"] = cluster_result
    result["dataset"] = dataset
    return result

def init_dataset(dataset):
    if dataset == "uci":
        train = uci_data
        test = uci_data
        test_lable = uci_target
        true_k = 10
        return train, test, test_lable, true_k
    elif dataset == "usps":
        train = usps_data
        test = usps_test
        test_lable = usps_test_lable
        true_k = 10
        return train, test, test_lable, true_k
    elif dataset == "mnist":
        train = mnist_train
        test = mnist_test
        test_lable = mnist_test_lable
        true_k = 10
        return train, test, test_lable, true_k
    elif dataset == "fashion_mnist":
        train = fashion_train
        test = fashion_test
        test_lable = fashion_test_lable
        true_k = 10
        return train, test, test_lable, true_k
    elif dataset == "cifar10":
        train = cifar_train
        test = cifar_test
        test_lable = cifar_test_lable_1d
        true_k = 10
        return train, test, test_lable, true_k

def store_in_mongo(collection, result):
    # init mongo client
    client = MongoClient("localhost",61003)
    db = client['scde_result']
    result_db = db[collection]

    # insert result to db
    result_db.insert_one(result)
    client.close()

def print_avg(collection, dataset):
    # init mongo client
    client = MongoClient("localhost",61003)
    db = client['scde_result']
    result_db = db[collection]

    results = result_db.find({"dataset":dataset})
    client.close()

    sum_de_sv_k = 0.0
    sum_sa_k = 0.0
    
    sum_k_nmi = 0.0
    sum_k_ari = 0.0

    sum_d_k_nmi = 0.0
    sum_d_k_ari = 0.0

    sum_d_sc_nmi = 0.0
    sum_d_sc_ari = 0.0

    sum_sc_nmi = 0.0
    sum_sc_ari = 0.0

    round = 0
    for result in results:
        round += 1

        estimate = result["estimate"]
        cluster_result = result['cluster_result']

        sum_de_sv_k += estimate["DE+SA"]
        sum_sa_k += estimate["SA"]

        sum_k_nmi += cluster_result["K-means"][0]
        sum_k_ari += cluster_result["K-means"][1]

        sum_d_k_nmi += cluster_result["DE+K-means"][0]
        sum_d_k_ari += cluster_result["DE+K-means"][1]

        sum_sc_nmi += cluster_result["SC"][0]
        sum_sc_ari += cluster_result["SC"][1]

        sum_d_sc_nmi += cluster_result["SCDE"][0]
        sum_d_sc_ari += cluster_result["SCDE"][1]

    print("times:",round)

    print("dataset:{}".format(dataset))
    
    print("DE+SA       | k:{}".format(sum_de_sv_k/round))
    print("SA          | k:{}".format(sum_sa_k/round))
    
    print()

    print("K-means     | NMI:{}".format(sum_k_nmi/round))
    print("DE, K-means | NMI:{}".format(sum_d_k_nmi/round))
    print("SC          | NMI:{}".format(sum_sc_nmi/round))
    print("SCDE        | NMI:{}".format(sum_d_sc_nmi/round))

    print()

    print("K-means     | ARI:{}".format(sum_k_ari/round))
    print("DE, K-means | ARI:{}".format(sum_d_k_ari/round))
    print("SC          | ARI:{}".format(sum_sc_ari/round))
    print("SCDE        | ARI:{}".format(sum_d_sc_ari/round))

# process
# for i in range(1):
    # store_in_mongo("first_5_round", process("uci"))
    # store_in_mongo("first_5_round", process("usps"))
    # store_in_mongo("first_5_round", process("mnist"))
    # store_in_mongo("first_5_round", process("fashion_mnist"))
    # store_in_mongo("first_5_round", process("cifar10"))

# g-means
# for i in range(5):
    # k = ek.estimate_gmeans(uci_data)
    # k = ek.estimate_gmeans(usps_test)
    # k = ek.estimate_gmeans(mnist_test)
    # k = ek.estimate_gmeans(fashion_test)
    # k = ek.estimate_gmeans(cifar_test)
    # print(k)