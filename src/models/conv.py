import numpy as np

class Conv3x3:
    """ A Convolution layer using 3x3 filters. """

    def __init__(self, num_filters):
        """ Create a new convolution layer """
        self.num_filters = num_filters

        # This is a 3-Tensor of num_filters * 3 * 3 convolution filters
        # Divide by nine to reduce initial variance - Xavier Initialization
        self.filters = np.random.randn(num_filters, 3, 3) / 9
        self.last_input = None

    def iterate_regions(self, image):
        """
        Generate 3x3 image regions using "valid" padding.

        Input:
        image - A 2D numpy array

        Yields a 3x3 image patch
        """
        h, w = image.shape
        patch_size = 3

        for i in range(h - (patch_size - 1)):
            for j in range(w - (patch_size - 1)):
                im_region = image[i:(i + patch_size), j:(j + patch_size)]
                yield im_region, i, j

    def forward(self, inp):
        """
        Perform a forward pass of the conv layer using the input image

        :param inp: A 2D numpy array
        :return: The convolved image - num_filters * (h - 2) * (w - 2)
        """
        h, w = inp.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        convolve = lambda mat, fil: np.sum(mat * fil, axis=(1, 2))

        for im_region, i, j in self.iterate_regions(inp):
            output[i, j] = convolve(im_region, self.filters)

        # Caching for backpropagation
        self.last_input = inp

        return output


    def backprop(self, d_L_d_out, eta):
        """
        Perform a backward phase on the conv layer so that we can update the filter weights

        :param d_L_d_out:
        :param eta:
        :return:
        """
        # Note that dL / d(filter(x, y)) = sum_i sum_j dL / d_out_ij * d_out_ij / d_filter_xy
        # Note also that d_out_ij / d_filter_xy = image(i + x, j + y)
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= eta * d_L_d_filters

        # We return None since we are using this as the first layer in our CNN
        # Otherwise we would need to return the loss gradient for this layer's inputs
        return None
