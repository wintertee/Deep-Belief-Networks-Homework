import cupy as np

import utils
from models import DNN

np.random.seed(0)

batch_size = 128
nb_iter_pretrain = 5
nb_iter_train = 10
lr_pretrain = 0.1
lr_train = 0.1


x, y = utils.lire_mnist_all("mnist_all.mat", train=True)
x = np.array(x)
y = np.array(y)

n_visible = [x.shape[1]]
n_hidden = [128, 128, 10]
dnn = DNN(n_visible + n_hidden)
dnn.pretrain(x, lr=lr_pretrain, batch_size=batch_size, nb_iter=nb_iter_pretrain)
print("Pretraining done.")
dnn.retropropagation(
    x, y, lr=lr_train, batch_size=batch_size, nb_iter=nb_iter_train, verbose=True
)

x, y = utils.lire_mnist_all("mnist_all.mat", train=False)
x = np.array(x)
y = np.array(y)
dnn.test(x, y, batch_size, verbose=True)
