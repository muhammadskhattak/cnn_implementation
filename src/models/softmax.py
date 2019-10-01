import numpy as np


class Softmax:
    """
    A standard fully-connected layer with softmax activation
    """

    def __init__(self, input_len, nodes):
        """
        Create a new Softmax model

        :param input_len: Size of the input (m x n x d)
        :param nodes: Number of output nodes
        """
        # Xavier Initialization
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)
        self.last_input_shape = None
        self.last_input = None
        self.last_totals = None

    def forward(self, inp):
        """
        Perform a forward pass of the softmax layer using the given
        input.

        :param inp: A 2d input image
        :return: A 1d numpy array of output probabilities
        """
        inp_flat = inp.flatten()

        input_len, nodes = self.weights.shape

        totals = np.dot(inp_flat, self.weights) + self.biases
        exp = np.exp(totals)

        # Cache these values for use in the back propagation phase
        self.last_input_shape = inp.shape
        self.last_input = inp_flat
        self.last_totals = totals

        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, eta):
        """
        Perform a backward pass on the softmax layer.
        The idea is that we will update based on this layers outputs and return the
        updated gradient for this layers inputs

        :param d_L_d_out: dL / dout - The loss gradient of this layers output
        :param eta: The learning rate for stochastic gradient descent
        :return: The updated gradient for this layer's input
        """
        # Only one term will be non-zero since the gradient of the output of the final predictive layer
        # will only be none zero for the ith-value, where out[i] = t[i] (target[i])
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)
            sum_totals = np.sum(t_exp)

            # Gradient with respect to the total sum
            d_out_d_t = -t_exp[i] * t_exp / (sum_totals ** 2)
            # Case where i == target
            d_out_d_t[i] = t_exp[i] * (sum_totals - t_exp[i]) / (sum_totals ** 2)

            # Note that t = w * input + b
            # Therefore dt / dw = input
            #           dt / db = 1
            #           dt / dinput = w
            # dL / dw = ( dL / dout ) * ( dout / dt ) * ( dt / dw )
            # dL / db = ( dL / dout ) * ( dout / dt ) * ( dt / db )
            # dL / dinput  = ( dL / dout ) * ( dout / dt ) * ( dt / dinput )

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradient of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradient of loss against weight, biases, input
            d_L_d_w =  d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]  # dimension input_len x nodes
            d_L_d_b = d_L_d_t * d_t_d_b                             # dimension nodes
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t                   # dimension input_len

            # Update weights/biases
            self.weights -= eta * d_L_d_w
            self.biases -= eta * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)
