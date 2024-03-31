from typing import List, Tuple

import cupy as np
from sklearn.utils import shuffle
from tqdm import tqdm

import functional as F


class RBM:
    """Restricted Boltzmann Machine (RBM) model."""

    def __init__(self, p: int, q: int):
        """
        Initialize the RBM.

        Parameters:
        p (int): The number of visible units.
        q (int): The number of hidden units.
        """
        self.p = p
        self.q = q

        # Initialize weights with normal distribution, mean=0, std=0.1 (variance=0.01)
        self.w = np.random.normal(0, 0.1, (p, q))

        # Initialize biases to zeros
        self.a = np.zeros(p)
        self.b = np.zeros(q)

    def entree_sortie(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probabilities of the hidden units given the visible units.

        Parameters:
        x (np.ndarray): A binary vector of length p representing
          the state of the visible units.

        Returns:
        np.ndarray: A binary vector of length q representing
          the probabilities of the hidden units.
        """
        return F.sigmoid(x @ self.w + self.b)

    def calcul_softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probabilities of the output units given the input units.

        Parameters:
        x (np.ndarray): A binary vector of length p representing
          the state of the visible units.

        Returns:
        np.ndarray: A binary vector of length q representing
          the probabilities of the output units.
        """
        return F.softmax(x @ self.w + self.b)

    def sortie_entree(self, h: np.ndarray) -> np.ndarray:
        """
        Compute the probabilities of the visible units given the hidden units.

        Parameters:
        h (np.ndarray): A binary vector of length q representing
          the state of the hidden units.

        Returns:
        np.ndarray: A binary vector of length p representing
          the probabilities of the visible units.
        """
        return F.sigmoid(h @ self.w.T + self.a)

    def train(
        self,
        x: np.ndarray,
        lr: float,
        batch_size: int,
        nb_iter: int,
        description: str = "",
        verbose: bool = True,
    ) -> float:
        """
        Train the RBM using the specified parameters.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.
        lr (float): The learning rate.
        batch_size (int): The size of the batches to be used in training.
        nb_iter (int): The number of iterations to train for.
        description (str): The description to show in the progress bar.

        Returns:
        float: The reconstruction error at the end of training.
        """

        assert lr > 0, "The learning rate must be positive."
        assert batch_size > 0, "The batch size must be positive."
        assert nb_iter > 0, "The number of iterations must be positive."

        description = f"Training {self}" if not description else description
        x = x.copy()
        n = x.shape[0]

        pbar = tqdm(range(nb_iter), desc=description, disable=not verbose)
        for i in pbar:
            # Shuffle the data
            np.random.shuffle(x)
            for j in range(0, n, batch_size):
                x_batch = x[j : min(n, j + batch_size)]
                tb = x_batch.shape[0]  # taille du batch

                # Compute the positive phase
                v_0 = x_batch  # tb x p
                p_h_v_0 = self.entree_sortie(v_0)  # tb x q
                h_0 = np.random.binomial(1, p_h_v_0) * 1  # tb x q
                p_v_h_0 = self.sortie_entree(h_0)  # tb x p
                v_1 = np.random.binomial(1, p_v_h_0) * 1  # tb x p
                p_h_v_1 = self.entree_sortie(v_1)  # tb x q

                # Compute the negative phase
                grad_a = np.sum(v_0 - v_1, axis=0)  # p
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)  # q
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                # Update the parameters
                self.w += lr * grad_w / tb
                self.a += lr * grad_a / tb
                self.b += lr * grad_b / tb

            # reconstruction mean squared error
            h = self.entree_sortie(x)
            x_recon = self.sortie_entree(h)
            mse = np.mean((x - x_recon) ** 2)

            pbar.set_postfix(reconstruction_mse=f"{mse:.4f}")

        return mse

    def generer_image(self, nb_iter_gibbs: int, nb_image: int):
        """
        Generate images using the RBM.

        Parameters:
        nb_iter_gibbs (int): The number of Gibbs sampling steps to perform.
        nb_image (int): The number of images to generate.

        Returns:
        np.ndarray: A binary matrix of size (nb_image, p) representing
          the generated images.
        """
        # Initialize the visible units
        # v = np.random.binomial(1, 0.5, (nb_image, self.p)) * 1

        # use random probabilities to initialize the visible units for each image
        v = np.zeros((nb_image, self.p), dtype=int)
        for i, p in enumerate(np.random.uniform(low=0, high=1, size=nb_image)):
            v[i, :] = np.random.binomial(1, p, self.p)

        # Perform Gibbs sampling
        for i in range(nb_iter_gibbs):
            p_h_v = self.entree_sortie(v)
            h = np.random.binomial(1, p_h_v) * 1
            p_v_h = self.sortie_entree(h)
            v = np.random.binomial(1, p_v_h) * 1

        return v

    def __str__(self):
        return f"RBM({self.p}, {self.q})"


class DBN:
    """Deep Belief Network (DBN) model."""

    def __init__(self, layers: list[int]):
        """
        Initialize the DBN.

        Parameters:
        layers (list[int]): The number of units in each layer.
        """
        self.layers = layers
        self.rbms = [RBM(p, q) for p, q in zip(layers[:-1], layers[1:])]

    def train(
        self,
        x: np.ndarray,
        lr: float,
        batch_size: int,
        nb_iter: int,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the DBN by successively training each RBM.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.
        lr (float): The learning rate.
        batch_size (int): The size of the batches to be used in training.
        nb_iter (int): The number of iterations to train for.
        verbose (bool): Whether to show process bar.

        Returns:
        List: A list of the mean squared errors of each RBM.

        """
        mse_list = []
        for i, rbm in enumerate(self.rbms):
            mse = rbm.train(
                x,
                lr,
                batch_size,
                nb_iter,
                description=f"Training {rbm} layer {i+1}/{len(self.rbms)}",
                verbose=verbose,
            )
            mse_list.append(mse)
            x = np.random.binomial(1, rbm.entree_sortie(x))

        return mse_list

    def generer_image(self, nb_iter_gibbs: int, nb_image: int):
        """
        Generate images using the DBN.

        Parameters:
        nb_iter_gibbs (int): The number of Gibbs sampling steps to perform.
        nb_image (int): The number of images to generate.

        Returns:
        np.ndarray: A binary matrix of size (nb_image, p) representing
          the generated images.
        """
        x = self.rbms[-1].generer_image(nb_iter_gibbs, nb_image)
        for rbm in self.rbms[-2::-1]:
            x = np.random.binomial(1, rbm.sortie_entree(x))
        return x


class DNN:
    """Deep Neural Network (DNN) model."""

    def __init__(self, layers: list[int]):
        """
        Initialize the DNN.

        Parameters:
        layers (list[int]): The number of units in each layer.
        """
        self.layers = layers
        self.dbn = DBN(layers[:-1])
        self.classifier = RBM(layers[-2], layers[-1])

    def pretrain(
        self,
        x: np.ndarray,
        lr: float,
        batch_size: int,
        nb_iter: int,
        verbose: bool = True,
    ) -> None:
        """
        Pretrain the DBN of the DNN using the specified parameters.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.
        lr (float): The learning rate.
        batch_size (int): The size of the batches to be used in training.
        nb_iter (int): The number of iterations to train for.
        verbose (bool): Whether to show process bar.

        """
        self.dbn.train(x, lr, batch_size, nb_iter, verbose=verbose)

    def entree_sortie_reseau(self, x: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        forward pass of the DNN.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.

        Returns:
        List: A list of hidden layers outputs.
        np.ndarray: A binary vector of length q representing
          the probabilities of the output units.
        """
        sorties = [x]
        for rbm in self.dbn.rbms:
            x = rbm.entree_sortie(x)
            sorties.append(x)
        probs = self.classifier.calcul_softmax(x)
        return sorties, probs

    def retropropagation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        batch_size: int,
        nb_iter: int,
        verbose: bool = True,
    ) -> float:
        """
        Train the DNN using the specified parameters.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.
        y (np.ndarray): A binary matrix of size (n, q) representing the labels.
        lr (float): The learning rate.
        batch_size (int): The size of the batches to be used in training.
        nb_iter (int): The number of iterations to train for.
        verbose (bool): Whether to show process bar.

        Returns:
        float: The accuracy of the model on the given data.

        """
        x = x.copy()
        n = x.shape[0]
        for i in range(nb_iter):
            # Shuffle the data
            x, y = shuffle(x, y)
            losses = []
            metrics = []

            pbar = tqdm(
                range(0, n, batch_size),
                desc=f"Iteration {i+1}/{nb_iter}",
                disable=not verbose,
            )
            for j in pbar:
                x_batch = x[j : min(n, j + batch_size)]
                y_batch = y[j : min(n, j + batch_size)]

                tb = x_batch.shape[0]  # taille du batch

                # forward pass
                sorties, probs = self.entree_sortie_reseau(x_batch)
                loss = F.cross_entropy(y_batch, probs)
                metric = F.accuracy(y_batch, probs)
                losses.append(loss)
                metrics.append(metric)

                # backward pass
                # classifier
                # print(probs.shape, y_batch.shape)
                grad_z = probs - y_batch
                # print(grad_z.shape)
                grad_w = sorties[-1].T @ grad_z
                # print(grad_w.shape)
                grad_b = np.sum(grad_z, axis=0)
                # print(grad_b.shape)
                grad_x = grad_z @ self.classifier.w.T
                # print(grad_x.shape)

                self.classifier.w -= lr * grad_w / tb
                self.classifier.b -= lr * grad_b / tb

                # dbn
                for k, rbm in enumerate(self.dbn.rbms[::-1]):
                    k = len(self.dbn.rbms) - k - 1  # k-th layer
                    layer_input = sorties[k]
                    layer_output = sorties[k + 1]

                    grad_a = grad_x  # a is the  activation
                    grad_z = grad_a * layer_output * (1 - layer_output)
                    grad_w = layer_input.T @ grad_z
                    grad_b = np.sum(grad_z, axis=0)
                    grad_x = grad_z @ rbm.w.T

                    rbm.w -= lr * grad_w / tb
                    rbm.b -= lr * grad_b / tb

                # compute the mean loss
                mean_loss = np.mean(np.array(losses)).item()
                mean_metric = np.mean(np.array(metrics)).item()
                pbar.set_postfix(loss=f"{mean_loss:.4f}", accuracy=f"{mean_metric:.4f}")

        return mean_metric

    def test(
        self, x: np.ndarray, y: np.ndarray, batch_size: int, verbose: bool = True
    ) -> float:
        """
        Test the DNN on the given data.

        Parameters:
        x (np.ndarray): A binary matrix of size (n, p) representing the training data.
        y (np.ndarray): A binary matrix of size (n, q) representing the labels.
        batch_size (int): The size of the batches to be used in training.
        verbose (bool): Whether to show process bar.

        Returns:
        float: The accuracy of the model on the given data.
        """
        n = x.shape[0]
        losses = []
        metrics = []
        pbar = tqdm(range(0, n, batch_size), disable=not verbose)
        for j in pbar:
            x_batch = x[j : min(n, j + batch_size)]
            y_batch = y[j : min(n, j + batch_size)]

            # forward pass
            _, probs = self.entree_sortie_reseau(x_batch)
            loss = F.cross_entropy(y_batch, probs)
            metric = F.accuracy(y_batch, probs)
            losses.append(loss)
            metrics.append(metric)

            # compute the mean loss
            mean_loss = np.mean(np.array(losses)).item()
            mean_metric = np.mean(np.array(metrics)).item()
            pbar.set_postfix(loss=f"{mean_loss:.4f}", accuracy=f"{mean_metric:.4f}")
        return mean_metric
