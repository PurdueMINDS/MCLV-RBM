import numpy as np
from PIL import Image as Image

from u.config import *
from u.log import Log
from u.utils import Util


class ImageService:
    @staticmethod
    def get_weights_as_images(W, num_hidden, width, height, out_dir=IMG_OUTPUT_FOLDER, index='0'):
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
            Util.tile_raster_images(
                X=weights.T,
                img_shape=(width, height),
                tile_shape=(num_hidden, 1),
                tile_spacing=(1, 1)
            )
        )
        image.save(out_dir + 'filters_at_epoch_%s.png' % str(index))

    @staticmethod
    def get_tours_as_images(num_hidden, iter, epoch, tile_length, img_dict
                            , width=28, height=28, out_dir=IMG_OUTPUT_FOLDER):
        for tour_type, tour_images in img_dict.items():
            Log.info("%s.shape=%s", tour_type, tour_images.shape)
            if tour_images.shape[0] != 0:
                tour_images = np.delete(tour_images, 0, axis=1)

                image = Image.fromarray(
                    Util.tile_raster_images(
                        X=tour_images,
                        img_shape=(width, height),
                        tile_shape=(tile_length, int(tour_images.shape[0] / tile_length)),
                        tile_spacing=(1, 1)
                    )
                )
                image.save(out_dir + '%s_h=%s,j=%s,epoch=%s.png' % (tour_type, num_hidden, iter, epoch))

    @staticmethod
    def get_components_as_images(components, name, width=28, height=28, out_dir=IMG_OUTPUT_FOLDER, tile_length=1):
        image = Image.fromarray(
            Util.tile_raster_images(
                X=components,
                img_shape=(width, height),
                tile_shape=(tile_length, int(components.shape[0] / tile_length)),
                tile_spacing=(1, 1)
            )
        )
        image.save(out_dir + name + '.png')

    @staticmethod
    def dump_tours(num_hidden, tour_lengths, history, cd_steps, iter, epoch):
        with open(IMG_OUTPUT_FOLDER + 'tour_lengths_h=%s' % num_hidden, 'a') as f:
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
