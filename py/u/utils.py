import hashlib
import json
import math
from numbers import Number

import numpy as np
import torch
from scipy import misc
from scipy import sparse

from b.supernode import SuperNode
from u.config import USE_DOUBLE
from u.gpu import Gpu


class MeanSD:
    def __init__(self, input_array, is_sub_array=False):
        self.is_sub_array = is_sub_array
        if not is_sub_array:
            self.mean = np.mean(input_array)
            self.sd = np.std(input_array)
        else:
            input_array = np.array(input_array)
            l0 = input_array.shape[0]
            ll = min([len(input_array[i]) for i in range(l0)])
            self.mean = [np.mean([input_array[j][i] for j in range(l0)]) for i in range(ll)]
            self.sd = [np.std([input_array[j][i] for j in range(l0)]) for i in range(ll)]

    def last_element(self):
        if self.is_sub_array:
            m, s = self.mean[len(self.mean) - 1], self.sd[len(self.sd) - 1]
        else:
            m, s = self.mean, self.sd
        return m, s

    @staticmethod
    def get(input):
        self = MeanSD(input)
        return (self.mean, self.sd)


class Util:
    @staticmethod
    def mle_discretize(matrix):
        matrix[matrix >= 0.5] = 1
        matrix[matrix < 0.5] = 0
        return matrix

    @staticmethod
    def compute_angle(a, b):
        angle = torch.sum(a * b) / (torch.norm(a, p=2) * torch.norm(b, p=2))
        return angle

    @staticmethod
    def dictize(**kwargs):
        return kwargs

    hash_mask_map = dict()

    @staticmethod
    def sum_ccdfs(a, b):
        if a is None:
            return b
        else:
            la = len(a)
            lb = len(b)
            a = np.lib.pad(a, (0, max(0, lb - la)), 'constant', constant_values=(0, 0))
            b = np.lib.pad(b, (0, max(0, la - lb)), 'constant', constant_values=(0, 0))
            return a + b

    @staticmethod
    def unique(input_array, gpu, supernode=False):
        nbits = input_array.shape[1]
        # To Handle torch's numerical instability we use exp
        converter = gpu.tensor_converter(gpu.tensor(nbits, 1).uniform_(1, 2))
        hashed = torch.mm(input_array, converter)
        hashed_np = hashed.cpu().numpy()
        _, idx = np.unique(hashed_np, True, axis=0)
        idx = gpu.from_numpy(idx)
        if supernode:
            return SuperNode(input_array[idx, :], hashed[idx], converter)
        else:
            return input_array[idx, :]

    @staticmethod
    def cartesian(x: np.ndarray, y: np.ndarray):
        return np.concatenate((np.tile(x, [len(y), 1]), np.repeat(y, len(x), axis=0)), axis=1)

    @staticmethod
    def scale_to_unit_interval(ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    @staticmethod
    def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.

        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).

        :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param X: a 2-D array in which every row is a flattened image.

        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image

        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)

        :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

        :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not


        :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
        :rtype: a 2-d array with same dtype as X.

        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape    = [0,0]
        # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [
            (ishp + tsp) * tshp - tsp
            for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
            ]

        if isinstance(X, tuple):
            assert len(X) == 4
            # Create an output np ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                     dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                     dtype=X.dtype)

            # colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    dt = out_array.dtype
                    if output_pixel_vals:
                        dt = 'uint8'
                    out_array[:, :, i] = np.zeros(
                        out_shape,
                        dtype=dt
                    ) + channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = Util.tile_raster_images(
                        X[i], img_shape, tile_shape, tile_spacing,
                        scale_rows_to_unit_interval, output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

            # generate a matrix to store the output
            dt = X.dtype
            if output_pixel_vals:
                dt = 'uint8'
            out_array = np.zeros(out_shape, dtype=dt)

            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        this_x = X[tile_col * tile_shape[0] + tile_row]
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = Util.scale_to_unit_interval(
                                this_x.reshape(img_shape))
                        else:
                            this_img = this_x.reshape(img_shape)
                        # add the slice to the corresponding position in the
                        # output array
                        c = 1
                        if output_pixel_vals:
                            c = 255
                        out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
            return out_array

    @staticmethod
    def sample_prob(probs, rand):
        """ Takes a tensor of probabilities (as from a sigmoidal activation)
        and samples from all the distributions

        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values

        :return : binary sample of probabilities
        """
        # return tf.nn.relu(tf.sign(probs - rand))
        return 0

    @staticmethod
    def gen_batches(data, batch_size):
        """ Divide input data into batches.

        :param data: input data
        :param batch_size: size of each batch

        :return: data divided into batches
        """
        data = np.array(data)

        for i in range(0, data.shape[0], batch_size):
            yield data[i:i + batch_size]

    @staticmethod
    def gen_image(img, width, height, outfile, img_type='grey'):
        assert len(img) == width * height or len(img) == width * height * 3

        if img_type == 'grey':
            misc.imsave(outfile, img.reshape(width, height))

        elif img_type == 'color':
            misc.imsave(outfile, img.reshape(3, width, height))

    @staticmethod
    def normalize(x):
        V = x.copy()
        V -= x.min(axis=1).reshape(x.shape[0], 1)
        V /= V.max(axis=1).reshape(x.shape[0], 1)
        return V

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def safe_divide(num, den):
        if den == 0:
            return den
        else:
            return num / den

    @staticmethod
    def ccdf(x):
        x = np.array(x)
        x_values = sorted(list(set(x)))
        y = np.array([float(len(x[x >= val])) for val in x_values])
        y = Util.safe_divide(y, sum(y))
        return x_values, y

    @staticmethod
    def empty(obj):
        return obj is None or len(obj) == 0

    @staticmethod
    def isnone(*args):
        """
        Returns the first non null element in the array.
        :param args:
        :return:
        """
        for obj in args:
            if obj is not None:
                return obj
        return None

    @staticmethod
    def add_bias_coefficient(an_array):
        if isinstance(an_array, sparse.csr_matrix):
            bias = sparse.csr_matrix(np.ones((an_array.shape[0], 1)))
            csr = sparse.hstack([bias, an_array]).tocsr()
        else:
            csr = np.insert(an_array, 0, 1, 1)
        return csr

    @staticmethod
    def shuffle(an_array):
        np.random.shuffle(an_array)
        return an_array

    @staticmethod
    def chunks(arr, step):
        arr = list(arr)
        l = len(arr)
        return [arr[i:min(i + step, l)] for i in range(0, l, step)]

    @classmethod
    def dict_to_json(cls, tour_lengths):
        return json.dumps(tour_lengths)

    @classmethod
    def json_to_dict(cls, tour_lengths):
        return json.loads(tour_lengths)

    @classmethod
    def put_or_add(cls, a_dict, key, value):
        if key not in a_dict:
            a_dict[key] = value
        else:
            a_dict[key] += value

    @classmethod
    def is_different(cls, h1, h2):
        return not (h1 == h2).all()

    @classmethod
    def printProgressBar(cls, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Credit:https://stackoverflow.com/users/2206251/greenstick
        Credit:https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    @classmethod
    def log_sum_exp(cls, a_list):
        x_max = np.max(a_list)
        diffs = a_list - x_max
        return x_max + np.log(np.sum(np.exp(diffs)))

    @classmethod
    def log_mean_exp(cls, a_list):
        return cls.log_sum_exp(a_list) - np.log(len(a_list))

    @classmethod
    def log_var_exp(cls, a_list):
        return np.log(np.exp(cls.log_mean_exp(a_list * 2)) - np.exp(2 * cls.log_mean_exp(a_list)))

    @classmethod
    def log_sd_exp(cls, a_list):
        return 0.5 * cls.log_var_exp(a_list)

    @staticmethod
    def md5(a_str):
        m = hashlib.md5()
        m.update(a_str.encode())
        return m.hexdigest()


if __name__ == '__main__':
    print(Util.isnone(None,1))
    print(Util.isnone(2,1))
    print(Util.isnone(3,2,1))
    print(Util.isnone(2,None,1))
    print(Util.isnone(None,None,None))
