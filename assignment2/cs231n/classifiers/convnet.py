import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class CustomConvNet(object):
  """
  A convolutional network with the following architecture:
  
  [conv - relu - 2x2 max pool]xM - [affine - relu]xN - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32], filter_size=3,
               hidden_dims=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dropout=0, use_batchnorm=True, dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: List of sizes of filters to use in each convolutional layer
    - hidden_dims: List of numbers of units to use in each fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all
    - use_batchnorm: Boolean; determines whether to use batch normalization
    - dtype: numpy datatype to use for computation
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.filter_size = filter_size

    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0

    self.num_C = len(num_filters)
    self.num_A = len(hidden_dims)
    
    """
    Initialize weights and biases.
    """
    
    C, H, W = input_dim

    ## Convolutional layers

    prev_depth = C

    for i in xrange(self.num_C):
        self.params['cW%d' % i] = np.random.randn(num_filters[i], prev_depth, filter_size, filter_size)
        self.params['cW%d' % i] *= weight_scale
        self.params['cB%d' % i] = np.zeros(num_filters[i])

        if self.use_batchnorm:
            self.params['cGamma%d' % i] = np.ones(num_filters[i])
            self.params['cBeta%d' % i] = np.zeros(num_filters[i])

        prev_depth = num_filters[i]

    ## Hidden layers

    prev_size = prev_depth * H/(2**(self.num_C/2)) * W/(2**(self.num_C/2))

    for i in xrange(self.num_A):
        self.params['aW%d' % i] = np.random.randn(prev_size, hidden_dims[i])
        self.params['aW%d' % i] *= weight_scale
        self.params['aB%d' % i] = np.zeros(hidden_dims[i])

        if self.use_batchnorm:
            self.params['aGamma%d' % i] = np.ones(hidden_dims[i])
            self.params['aBeta%d' % i] = np.zeros(hidden_dims[i])

        prev_size = hidden_dims[i]

    # Output layer
    self.params['sW'] = np.random.randn(prev_size, num_classes) * weight_scale
    self.params['sB'] = np.zeros(num_classes)

    # Dropout parameteres
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # Batch normalization parameters
    self.bn_params = {'c': [], 'a': []}
    if self.use_batchnorm:
      self.bn_params['c'] = [{'mode': 'train'} for i in xrange(self.num_C)]
      self.bn_params['a'] = [{'mode': 'train'} for i in xrange(self.num_A)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the convolutional network.
    """
    mode = 'test' if y is None else 'train'
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode 
    if self.use_batchnorm:
      for bn_param in self.bn_params['c'] + self.bn_params['a']:
        bn_param[mode] = mode

    """
    Forward pass
    """
    caches = []
    out = X

    # Convolutional layers
    for i in xrange(self.num_C):
        cW, cB = self.params['cW%d' % i], self.params['cB%d' % i]
        if i % 2 == 0:
            out, c = conv_relu_forward(out, cW, cB, conv_param)
        else:
            out, c = conv_relu_pool_forward(out, cW, cB, conv_param, pool_param)
        caches.append(c)

        if self.use_batchnorm:
            cGamma, cBeta = self.params['cGamma%d' % i], self.params['cBeta%d' % i]
            out, c = spatial_batchnorm_forward(out, cGamma, cBeta, self.bn_params['c'][i])
            caches.append(c)

    # Hidden layers
    for i in xrange(self.num_A):
        aW, aB = self.params['aW%d' % i], self.params['aB%d' % i]
        out, c = affine_relu_forward(out, aW, aB)
        caches.append(c)

        if self.use_dropout:
            out, c = dropout_forward(out, self.dropout_param)
            caches.append(c)

        if self.use_batchnorm:
            aGamma, aBeta = self.params['aGamma%d' % i], self.params['aBeta%d' % i]
            out, c = batchnorm_forward_alt(out, aGamma, aBeta, self.bn_params['a'][i])
            caches.append(c)

    # Output layer; scores
    sW, sB = self.params['sW'], self.params['sB']
    scores, c = affine_forward(out, sW, sB)
    caches.append(c)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    # Compute scores loss, gradient
    loss, dscores = softmax_loss(scores, y)

    # Regularization loss
    for param_name in self.params.keys():
        if 'W' in param_name:
            W = self.params[param_name]
            loss += 0.5 * self.reg * np.sum(W**2)

    ## Backpropagation
    
    # Output layer
    dout, dsW, dsB = affine_backward(dscores, caches.pop())
    grads['sB'], grads['sW'] = dsB, dsW + self.reg * sW

    # Hidden layers
    for i in reversed(xrange(self.num_A)):
        if self.use_batchnorm:
            dout, daGamma, daBeta = batchnorm_backward_alt(dout, caches.pop())
            grads['aGamma%d' % i], grads['aBeta%d' % i] = daGamma, daBeta

        if self.use_dropout:
            dout = dropout_backward(dout, caches.pop())

        dout, daW, daB = affine_relu_backward(dout, caches.pop())
        grads['aB%d' % i], grads['aW%d' % i] = daB, daW + self.reg * self.params['aW%d' % i]

    # Convolutional layerss
    for i in reversed(xrange(self.num_C)):
        if self.use_batchnorm:
            dout, dcGamma, dcBeta = spatial_batchnorm_backward(dout, caches.pop())
            grads['cGamma%d' % i], grads['cBeta%d' % i] = dcGamma, dcBeta

        if i % 2 == 0:
            dout, dcW, dcB = conv_relu_backward(dout, caches.pop())
        else:
            dout, dcW, dcB = conv_relu_pool_backward(dout, caches.pop())
        grads['cB%d' % i], grads['cW%d' % i] = dcB, dcW + self.reg * self.params['cW%d' % i]
    
    return loss, grads
  
pass
