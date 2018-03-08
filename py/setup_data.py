import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from u.config import MNIST_FOLDER

np.set_printoptions(threshold=0, suppress=True)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, _, test, _ = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    np.save(MNIST_FOLDER + "data.npy", data)
    np.save(MNIST_FOLDER + "test.npy", test)
