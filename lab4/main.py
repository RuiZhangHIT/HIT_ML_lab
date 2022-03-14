import numpy as np

import mytool as mt
import pca


if __name__ == '__main__':
    np.random.seed(0)
    mus = [[2, 4, 6]]
    sigmas = [[1, 1, 0.1]]
    nums = [100]
    data = mt.generate_data(mus, sigmas, nums, dim=3)
    rotate_data = mt.rotate(data.T).T
    compressed_data, reconstructed_data = pca.pca(rotate_data, 2)
    mt.show_pca_result(rotate_data, reconstructed_data)

    data = mt.load_faces()
    for k in [10, 5, 3, 1]:
        compressed_data, reconstructed_data = pca.pca(data, k)
        mt.show_compressed_faces(data, reconstructed_data)
