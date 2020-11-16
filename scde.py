import tensorflow.keras.backend as K

# standard squered error is a loss function
def sse(y_true, y_pred):
    return K.sum(K.square(y_pred-y_true),axis=-1) # TODO what is -1?

# z: the output of inner layer
def loss_func_in_scde_plus(z):
    def loss(y_true, y_pred):
        return sse(y_true,y_pred) - K.sum(K.square(z),axis=-1)
    return loss