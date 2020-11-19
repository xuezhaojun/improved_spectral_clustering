# provide datasets for clustering
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
import h5py
from tensorflow.keras.datasets import reuters,cifar100
from sklearn.datasets import fetch_20newsgroups

# minist
def get_mnist():
    (train, train_lable), (test, test_lable) = mnist.load_data()
    return train,train_lable, test, test_lable

# fashion-mnist
def get_fashion_mnist():
    (train,train_lable),(test,test_lable) = fashion_mnist.load_data()
    return train, train_lable, test, test_lable

# usps
# with h5py.File(path, 'r') as hf:
#         train = hf.get('train')
#         X_tr = train.get('data')[:]
#         y_tr = train.get('target')[:]
#         test = hf.get('test')
#         X_te = test.get('data')[:]
#         y_te = test.get('target')[:]
def get_usps():
    path = "./datasets/usps.h5"
    with h5py.File(path,"r") as hf:
        train = hf.get('train')
        data = train.get('data')[:]
        target = train.get('target')[:]
        return data, target

# cifar20
def get_cifar20():
    (train, train_label), (test, test_lables) = cifar100.load_data(label_mode="coarse")
    return train,train_label,test,test_lables

# reuters-8
def get_reuters_8():
    (train, train_label), (test, test_lables) = reuters.load_data()
    return train,train_label,test,test_lables

# 20 newsgroups
def get_20_newsgroups():
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None
    print("Loading 20 newsgroups dataset for categories:")
    print(categories) # 如果需要全部的类目，则制定为none即可

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
    return dataset