import numpy as np
from src.models.conv import Conv3x3
from src.models.maxpool import MaxPool2
from src.models.softmax import Softmax


class CNN:
    """ A CNN model """

    def __init__(self):
        self.conv = Conv3x3(8)  # 28 x 28 x 1 -> 26 x 26 x 8
        self.pool = MaxPool2()  # Pooling layer
        self.softmax = Softmax(13 * 13 * 8, 10)  # Softmax predictor

    def forward(self, image, label):
        """
        Perform a forward pass of this CNN using a single image

        :param image: A 2D input
        :param label: The target for the input
        :return: probability vector, loss and indicator variable
        """
        # Transform the image to be in range [-0.5, 0.5] (more "normal" range)
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        # Cross entropy loss
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, image, label, eta=0.005):
        """
        Train the model stochastically on the given image and label

        :param image: A 2D image
        :param label: the target output
        :param eta: the learning rate
        :return: loss, acc
        """
        # Forward pass
        out, loss, acc = self.forward(image, label)

        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # Backpropagation
        gradient = self.softmax.backprop(gradient, eta)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, eta)

        return loss, acc
