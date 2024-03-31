from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def lire_alpha_digit(dataset_path: str, characters: List[int]) -> np.array:
    """
    Load the BinaryAlphaDigits dataset and extract the data for specified characters.

    Parameters:
    characters (list of int): A list of integers representing
      the characters to be loaded (e.g., [10, 11, 12] for A, B, C).
    dataset_path (str): The path to the BinaryAlphaDigits.mat file.

    Returns:
    np.ndarray: A matrix where each row is a flattened image of
      the specified characters.
    """
    # Load the dataset
    data = loadmat(dataset_path)["dat"]

    # Extract the specified characters
    data = data[characters, :]

    # Flatten and stack the data
    data = [img.flatten() for char in data for img in char]
    data = np.vstack(data)

    return data


def lire_mnist(dataset_path: str, characters: List[str]) -> np.array:
    """
    Load the MNIST dataset and extract the data for specified characters.

    Parameters:
    characters (list of str): A list of strings representing
      the characters to be loaded (e.g., ["0", "1", "2"] for 0, 1, 2).
    dataset_path (str): The path to the MNIST.mat file.

    Returns:
    np.ndarray: A matrix where each row is a flattened image of
      the specified characters.

    """
    data = loadmat(dataset_path)
    all_data = []

    for key in data.keys():
        if key in characters:
            dataset = data[key]
            all_data.append(dataset)

    data = np.concatenate(all_data)
    threshold = 128
    return (data > threshold).astype(np.uint8)


def lire_mnist_all(dataset_path: str, train: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset and extract the data for specified characters.

    Parameters:
    train (bool): Whether to load the training or testing data.
    dataset_path (str): The path to the MNIST.mat file.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the input and output data.

    """
    data = loadmat(dataset_path)
    num_classes = 10
    x, y = None, None

    for num in range(num_classes):
        key = f"train{num}" if train else f"test{num}"
        dataset = data[key]
        labels = np.full((dataset.shape[0],), num, dtype=np.uint8)

        if x is None:
            x = dataset
            y = labels
        else:
            x = np.concatenate((x, dataset))
            y = np.concatenate((y, labels))

    y = np.eye(num_classes, dtype=np.uint8)[y]

    threshold = 128
    x = (x > threshold).astype(np.uint8)

    return x, y


def show_images(data: np.ndarray, n_cols: int = 10, height: int = 20, width: int = 16):
    """
    Display the specified images.

    Parameters:
    data (np.ndarray): A matrix where each row is a flattened image.
    n_cols (int): The number of columns to use in the display.
    height (int): The height of the images.
    width (int): The width of the images.
    """
    n_images = data.shape[0]
    n_rows = int(np.ceil(n_images / n_cols))

    data = data.reshape((n_images, height, width)).get()  # to host momery
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(1 - data[i], cmap="gray")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
