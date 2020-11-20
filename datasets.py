# provide datasets for clustering
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
import h5py
from tensorflow.keras.datasets import cifar100,cifar10
from sklearn import datasets

# datasets sorted by dim

# uci
# This is a copy of the test set of the UCI ML hand-written digits datasets
# dim: 64
# train: 1797
# class: 10
def get_uci():
    uci = datasets.load_digits()
    return uci.data, uci.target

# usps
# dim: 256
# train: 7291
# test: 2007
# class: 10
def get_usps():
    path = "./datasets/usps.h5"
    with h5py.File(path,"r") as hf:
        train = hf.get('train')
        data = train.get('data')[:]
        target = train.get('target')[:]

        test = hf.get('test')
        test_data = test.get('data')[:]
        test_target = test.get('target')[:]
        return data, target, test_data, test_target

# minist
# dim: 784
# train: 60000
# test: 10000
# class: 
def get_mnist():
    (train, train_lable), (test, test_lable) = mnist.load_data()
    return train,train_lable, test, test_lable

# fashion-mnist
# dim: 784
# train: 60000
# test: 10000
# class: 10
def get_fashion_mnist():
    (train,train_lable),(test,test_lable) = fashion_mnist.load_data()
    return train, train_lable, test, test_lable

# cifar10
# dim: 3072
# train: 50000
# test: 10000
# class: 10
def get_cifar10():
    (train, train_lable), (test, test_lables) = cifar10.load_data()
    return train, train_lable, test, test_lables