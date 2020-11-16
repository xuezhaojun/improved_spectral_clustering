# provide datasets for clustering
import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.datasets.fashion_mnist as fashion_mnist
import h5py
from tensorflow.keras.datasets import reuters
from sklearn.datasets import fetch_20newsgroups

# minist
def get_mnist():
    (x_train, lable), (_, _) = mnist.load_data()
    return x_train,lable

# fashion-mnist
def get_fashion_mnist():
    (x_train,lable),(_,_) = fashion_mnist.load_data()
    return x_train, lable

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

# stl-10
# TODO：size = 2GB so not going to prepare this one
# the python code to read the data : https://github.com/mttk/STL10/blob/master/stl10_input.py
def get_stl_10():
    return

# reuters-8
def get_reuters_8():
    (train_data, train_labels), (_, _) = reuters.load_data()
    return train_data,train_labels

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