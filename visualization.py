import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector

from utils import mnist_reader
from utils.helper import get_sprite_image
from configs import DATA_DIR, VIS_DIR
import os


class FashionMNISTVisualizer:
    """
    Provides interfaces for visualizing data in the Fashion_MNIST data set.
    """

    Y_LABELS = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    TENSORFLOW_VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensorvis')
    HEIGHT = 28
    WIDTH = 28

    def __init__(self, x=None, y=None):
        """
        @param x - MxN matrix with M data points and N features.
        @param y - Mx1 matrix with M data points containing a classification value for each data point in X.
        """
        self._X = x
        self._Y = y

    def load_mnist_data(self, data_path, kind):
        """
        @param path - Path to directory containing Fashion-MNIST data. (Use variables from configs.py)
        @param kind - Which dataset to load. ("t10k" or "train")
        """
        self._X, self._Y = mnist_reader.load_mnist(path=data_path, kind=kind)

    def data_dimenions(self):
        """
        @return - Two tuples. The first tuple is the dimensionality of X, the second is the dimensionality of Y.
        """
        return self._X.shape, self._Y.shape

    def y_values(self):
        """
        @return - A NumPy ndarray of unique Y values. (e.g. [0, 1, 2, 3, 4])
        """
        return np.unique(self._Y)

    def generate_tensorflow_files(self, dir=TENSORFLOW_VIS_DIR):
        """
        @param dir - Location where the Tensorflow files will be saved to.
        """
        y_str = np.array([FashionMNISTVisualizer.Y_LABELS[j] for j in self._Y])
        np.savetxt(os.path.join(dir, 'Xtest.tsv'), self._X, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(dir, 'Ytest.tsv'), y_str, fmt='%s')
        plt.imsave(os.path.join(dir, 'zalando-mnist-sprite.png'), get_sprite_image(self._X), cmap='gray')

    def generate_tensorboard_files(self, dir=TENSORFLOW_VIS_DIR):
        """
        Run `tensorboard --logdir=TENSORFLOW_VIS_DIR --host localhost` to host local instance
        after these files are generated.

        @param dir - Location where Tensorboard files will be saved to. These are used by
        the Tensorboard Projector.
        """
        embedding_var = tf.Variable(self._X, name='mnist_pixels')
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # Link this tensor to its metadata file (e.g. labels).

        embedding.metadata_path = os.path.join(dir, 'Ytest.tsv')
        embedding.sprite.image_path = os.path.join(dir, 'zalando-mnist-sprite.png')
        embedding.sprite.single_image_dim.extend([FashionMNISTVisualizer.WIDTH, FashionMNISTVisualizer.HEIGHT])

        summary_writer = tf.summary.FileWriter(dir)
        projector.visualize_embeddings(summary_writer, config)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir, 'model.ckpt'), 0)


if __name__ == "__main__":
    vis = FashionMNISTVisualizer()
    vis.load_mnist_data(data_path=DATA_DIR, kind='t10k')
    vis.generate_tensorflow_files()
    vis.generate_tensorboard_files()
