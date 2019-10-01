import numpy as np


class MaxPool2:
    """ A class that max pools with a pool size of 2 """
    def __init__(self):
        self.last_input = None

    def iterate_regions(self, image):
        """
        Generate a non-overlapping 2x2 image regions to pool over

        :param image: a 2D numpy array

        Yields regions
        """
        h, w, n = image.shape
        pool_h = h // 2
        pool_w = w // 2

        for i in range(pool_h):
            for j  in range(pool_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, inp):
        """
        Pool the input using a 2x2 max filter
        """
        h, w, num_filters = inp.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(inp):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        # Caching for use in backpropagation
        self.last_input = inp

        return output

    def backprop(self, d_L_d_out):
        """
        Perform a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs

        The idea of this layer is that since the forward propagation reduced
        the input size by half we want to double it ie produce a 4x4 filter
        that maps to each max argument. ie
        [                   [
                                [0, 0, 2, 0]
            [1, 2]      =>      [0, 1, 0, 0]
            [3, 4]              [3, 0, 0, 4]
                                [0, 0, 0, 0]
        ]                   ]

        :param d_L_d_out: The loss w.r.t the output layer
        :return:
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
