import math
import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        num_filters_1=6,
        num_filters_2=16,
        filter_size=5,
        hidden_dim=100,
        num_classes=10,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        k = 1 / (self.C * self.filter_size**2)
        self.params["W1"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            (self.num_filters_1, self.C, self.filter_size, self.filter_size),
        )
        # after first conv we sum over channels, and the number of filters in first conv gives the
        # 2nd dimension for weights in second conv in place of # channels
        self.params["W2"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            (
                self.num_filters_2,
                self.num_filters_1,
                self.filter_size,
                self.filter_size,
            ),
        )
        # dimensions after conv filtering are N, F, self.H - filter_size + 1, self.W - filter_size + 1
        # dimensions after pooling are (h - f + 1) / s x (w - f + 1) / s,
        # where f is the filter size of of the pooling layer and s is the stride.
        # f = 2, s = 2
        # H_prime_1 and W_prime_1 are dimensions after applying 1 conv filter and max pooling
        # H_primw_2 and W_prime_2 after second conv filter and max pooling
        # W3 has dimensions num_filters_2 x (H_prime_2 * W_prime_2) x hidden_dim,
        # because of flattening after pooling
        H_prime_1 = math.floor((self.H - filter_size + 1) / 2)
        W_prime_1 = math.floor((self.W - filter_size + 1) / 2)
        H_prime_2 = math.floor((H_prime_1 - filter_size + 1) / 2)
        W_prime_2 = math.floor((W_prime_1 - filter_size + 1) / 2)
        k = 1 / (self.num_filters_2 * H_prime_2 * W_prime_2)
        self.params["W3"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            (self.num_filters_2 * H_prime_2 * W_prime_2, self.hidden_dim),
        )
        self.params["b3"] = np.random.uniform(
            -np.sqrt(k), np.sqrt(k), (self.hidden_dim)
        )
        k = 1 / (self.num_filters_2 * self.hidden_dim * self.hidden_dim)
        self.params["W4"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            (self.hidden_dim, self.num_classes),
        )
        self.params["b4"] = np.random.uniform(
            -np.sqrt(k), np.sqrt(k), (self.num_classes)
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        conv1, conv1_cache = conv_forward(X, W1)
        # relu
        relu1, relu1_cache = relu_forward(conv1)
        pool1, pool1_cache = max_pool_forward(relu1, pool_param)
        conv2, conv2_cache = conv_forward(pool1, W2)
        relu2, relu2_cache = relu_forward(conv2)
        pool2, pool2_cache = max_pool_forward(relu2, pool_param)

        # flatten pool2 before fc layer
        flattened = pool2.reshape(pool2.shape[0], -1)

        fc1, fc1_cache = fc_forward(flattened, W3, b3)
        relu3, relu3_cache = relu_forward(fc1)
        fc2, fc2_cache = fc_forward(relu3, W4, b4)
        scores = fc2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        loss_update, grads_scores = softmax_loss(scores, y)
        loss += loss_update
        grads_fc2 = fc_backward(grads_scores, fc2_cache)
        _, grads["W4"], grads["b4"] = grads_fc2
        grads_relu3 = relu_backward(grads_fc2[0], relu3_cache)
        grads_fc1 = fc_backward(grads_relu3, fc1_cache)
        _, grads["W3"], grads["b3"] = grads_fc1
        grads_pool2 = max_pool_backward(grads_fc1[0].reshape(pool2.shape), pool2_cache)
        grads_relu2 = relu_backward(grads_pool2, relu2_cache)
        grads_conv2 = conv_backward(grads_relu2, conv2_cache)
        _, grads["W2"] = grads_conv2
        grads_pool1 = max_pool_backward(grads_conv2[0], pool1_cache)
        grads_relu1 = relu_backward(grads_pool1, relu1_cache)
        grads_conv1 = conv_backward(grads_relu1, conv1_cache)
        _, grads["W1"] = grads_conv1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
