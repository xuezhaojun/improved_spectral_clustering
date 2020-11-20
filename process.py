import numpy as np
import clusters
import datasets as ds
import estimate_k as ek
import gmeans

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

def process(dataset, train, test, test_lable, true_k):
    estimate = {}
    cluster_result = {}

    # DE
    de_result = clusters.de(train,test)

    # --- estimate k ---

    # DE+SA
    k = ek.get_k_from_result(ek.estimate(2*true_k, de_result))
    estimate["DE+SA"] = k

    # SA
    k = ek.get_k_from_result(ek.estimate(2*true_k,test))
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

    # init mongo client
    client = MongoClient("localhost",61003)
    db = client['scde_result']
    result_db = db.first_10_round

    # insert result to db
    result_db.insert_one(result)


process("mnist", mnist_train, mnist_test, mnist_test_lable,10)

for i in range(0):
    process("uci",uci_data,uci_data,uci_target,10)
    process("usps",usps_data,usps_test, usps_test_lable,10)
    process("mnist", mnist_train, mnist_test, mnist_test_lable,10)
    process("fashion_mnist",fashion_train, fashion_test, fashion_test_lable,10)
    process("cifar10", cifar_train, cifar_test, cifar_test_lable_1d, 10)