# provide datasets for clustering
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
import h5py
from tensorflow.keras.datasets import cifar10
from sklearn.datasets import fetch_20newsgroups
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords # interesting 'This' is not a stopwords but 'this' is
from nltk.stem.porter import PorterStemmer

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

# 20newsgroups
# dim: 6336
# train: 11314
# test: 7532
# TODO for now we reture 2000 features but not the most frequent words
def get_20_newsgroups(): 
    vectorizer = TfidfVectorizer(stop_words='english', min_df=30, max_df=0.2)
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    train = vectorizer.fit_transform(train_data.data)
    train = train.toarray()
    test = vectorizer.fit_transform(test_data.data)
    test = test.toarray()
    return train, train_data.target, test, test_data.target