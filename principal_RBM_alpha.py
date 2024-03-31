import cupy as np

import utils
from models import RBM

np.random.seed(0)

characters = [10, 11, 12]
data = utils.lire_alpha_digit("binaryalphadigs.mat", characters)
data = np.array(data)

n_visible = data.shape[1]
n_hidden = 128
rbm = RBM(n_visible, n_hidden)
score = rbm.train(data, lr=0.1, batch_size=16, nb_iter=200, verbose=True)

print(f"Score: {score:.4f}")
utils.show_images(rbm.generer_image(nb_iter_gibbs=200, nb_image=10), 10)
