import cupy as np

import utils
from models import DBN

np.random.seed(0)

characters = [10, 11, 12]
data = utils.lire_alpha_digit("binaryalphadigs.mat", characters)
data = np.array(data)

n_visible = [data.shape[1]]
n_hidden = [128] * 2
dbn = DBN(n_visible + n_hidden)
dbn.train(data, lr=0.1, batch_size=16, nb_iter=200)

utils.show_images(dbn.generer_image(nb_iter_gibbs=100, nb_image=10), 10)
