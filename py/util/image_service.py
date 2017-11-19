# Copyright 2017 Bruno Ribeiro, Mayank Kakodkar, Pedro Savarese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from PIL import Image as Image

from util.config import *
from util.log import Log
from util.utils import Util


class ImageService:
    @classmethod
    def get_weights_as_images(cls, W, num_hidden, width, height, out_dir=OUTPUT_FOLDER, index='0'):
        """
        Create and save the weights of the hidden units with respect to the
        visible units as images.

        Basically visualize the visible state as a sum of the hidden layers weighted by the hidden state

        :param index:
        :param width:
        :param height:
        :param out_dir:

        :return: self
        """

        weights = W[1:, 1:]

        image = Image.fromarray(
            cls.tile_raster_images(
                X=weights.T,
                img_shape=(width, height),
                tile_shape=(num_hidden, 1),
                tile_spacing=(1, 1)
            )
        )
        image.save(out_dir + 'filters_at_epoch_%s.png' % str(index))

    @classmethod
    def get_tours_as_images(cls, num_hidden, iter, epoch, tile_length, img_dict
                            , width=28, height=28, out_dir=OUTPUT_FOLDER):
        for tour_type, tour_images in img_dict.items():
            Log.info("%s.shape=%s", tour_type, tour_images.shape)
            if tour_images.shape[0] != 0:
                tour_images = np.delete(tour_images, 0, axis=1)

                image = Image.fromarray(
                    cls.tile_raster_images(
                        X=tour_images,
                        img_shape=(width, height),
                        tile_shape=(tile_length, int(tour_images.shape[0] / tile_length)),
                        tile_spacing=(1, 1)
                    )
                )
                image.save(out_dir + '%s_h=%s,j=%s,epoch=%s.png' % (tour_type, num_hidden, iter, epoch))

    @classmethod
    def get_components_as_images(cls, components, name, width=28, height=28, out_dir=OUTPUT_FOLDER, tile_length=1):
        image = Image.fromarray(
            cls.tile_raster_images(
                X=components,
                img_shape=(width, height),
                tile_shape=(tile_length, int(components.shape[0] / tile_length)),
                tile_spacing=(1, 1)
            )
        )
        image.save(out_dir + name + '.png')

    @staticmethod
    def dump_tours(num_hidden, tour_lengths, history, cd_steps, iter, epoch):
        with open(OUTPUT_FOLDER + 'tour_lengths_h=%s' % num_hidden, 'a') as f:
            f.write(" ".join(map(lambda x: str(int(x)), tour_lengths)))
            f.write(" ")

        short_tour = []
        inter_tour = []
        long_tour = []

        idxs = [0, 1, -1]

        for tour in history:
            v = [s.v for s in tour]
            Log.var(len_v_history=len(v))
            if len(v) == 2:
                short_tour += list(v[i] for i in idxs)
            elif len(v) == cd_steps + 1:
                long_tour += list(v[i] for i in idxs)
            else:
                inter_tour += list(v[i] for i in idxs)

        ImageService.get_tours_as_images(num_hidden, iter, epoch, len(idxs)
                                         , Util.dictize(short_tour=np.array(short_tour)
                                                        , inter_tour=np.array(inter_tour)
                                                        , long_tour=np.array(long_tour))
                                         )

    @classmethod
    def tile_raster_images(cls, X, img_shape, tile_shape, tile_spacing=(0, 0),
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
                    out_array[:, :, i] = cls.tile_raster_images(
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
