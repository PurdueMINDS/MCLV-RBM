import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from u.config import IMG_OUTPUT_FOLDER, MNIST_FOLDER
from u.image_service import ImageService


# plt.style.use('classic')
# font = {'size': 21}
#
# plt.rc('font', **font)




# print(plt.style.available)



class PcaService:
    font_scale = 1.75
    line_styles = [
        [], [3, 6, 3, 6, 3, 18], [12, 6, 12, 6, 3, 6], [12, 6, 3, 6, 3, 6]
    ]

    @classmethod
    def plot(cls, data, ica=True, n_components=5):
        if ica:
            method = FastICA
            method_name = "ICA"
        else:
            method = PCA
            method_name = "PCA"

        model = method(n_components=n_components)
        u = model.fit_transform(data)
        for i in range(1, n_components):
            for j in range(0, i):
                plt.clf()
                plt.scatter(u[:, j], u[:, i])
                plt.savefig(IMG_OUTPUT_FOLDER + "%s_%s_%s.png" % (method_name, j, i))
        ImageService.get_components_as_images(model.components_, method_name + "_images")


if __name__ == '__main__':
    data = np.load(MNIST_FOLDER + "data.npy")
    PcaService.plot(data, False, 2)
