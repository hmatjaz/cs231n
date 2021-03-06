import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # Implement the affine forward pass. Store the result in out. You           #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  
  # Dimensions
  N = x.shape[0]
  D = w.shape[0]

  # Reshape input and perform affine transform
  out = x.reshape(N, D).dot(w) + b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # Implement the affine backward pass.                                       #
  #############################################################################
  
  # Dimensions
  N = x.shape[0]
  D = w.shape[0]

  # Gradient with respect to bias is a column-wise sum
  db = np.sum(dout, axis=0)

  # Gradient with respect to dx and dw is multiplication with other factor,
  # but some reshaping is required on x and dx
  dw = x.reshape(N, D).T.dot(dout)
  dx = dout.dot(w.T).reshape(*x.shape) 

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # Implement the ReLU forward pass.                                          #
  #############################################################################
  
  out = np.maximum(0, x)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # Implement the ReLU backward pass.                                         #
  #############################################################################
  
  # Explicitly copying is a safety practice so we don't overwrite our inputs.
  dx = dout.copy()

  # No gradient flow where x was 0 or less
  dx[x <= 0] = 0

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # Implement the training-time forward pass for batch normalization.         #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    # Compute mean
    x_sum = np.sum(x, axis=0)
    mean = x_sum / N

    # Compute standard deviation
    centered_x = x - mean
    centered_x_sq = centered_x ** 2
    centered_x_sq_sum = np.sum(centered_x_sq, axis=0)
    variance = centered_x_sq_sum / N
    std = np.sqrt(variance)

    # Update running stats
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * std

    # Normalize
    normal = centered_x / (std + eps)

    # Apply scale and shift
    out = normal * gamma + beta

    cache = (x, bn_param, gamma, x_sum, mean, centered_x, centered_x_sq, centered_x_sq_sum, variance, std, normal)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # Implement the test-time forward pass for batch normalization. Use         #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    # Normalize
    out = (x - running_mean) / (running_var + eps)

    # Apply scale and shift
    out *= gamma
    out += beta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # Implement the backward pass for batch normalization. Store the            #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  
  # Unpack the cache
  x, bn_param, gamma, x_sum, mean, centered_x, centered_x_sq, centered_x_sq_sum, variance, std, normal = cache
  N = x.shape[0]
  eps = bn_param.get('eps', 1e-5)  # Very important for numerical stability!

  # Compute gradients on beta and gamma
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(normal * dout, axis=0)

  # Gradient on normalized X
  dnormal = dout * gamma

  # Gradient on standard deviation
  dstd = -1.0 * np.sum(centered_x * dnormal, axis=0) / (std ** 2 + eps)

  # Gradient on centered X
  dcentered_x = dnormal / (std + eps)

  # Gradient on variance
  dvariance = (0.5 / (std + eps)) * dstd

  # Gradient oo sum of squares
  dcentered_x_sq_sum = dvariance / N

  # Gradient on squares
  dcentered_x_sq = np.ones_like(x) * dcentered_x_sq_sum 

  # Gradient on centered X (note the addition)
  dcentered_x += 2.0 * centered_x * dcentered_x_sq

  # Gradient on mean
  dmean = -1.0 * np.sum(dcentered_x, axis=0)

  # Gradient on X sum
  dx_sum = dmean / N

  # Gradient on X
  dx = np.zeros_like(x)
  dx += dcentered_x
  dx += np.ones_like(x) * dx_sum

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

def batchnorm_forward_alt(x, gamma, beta, bn_param):
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None

  ## Training
  if mode == 'train':
    # Compute mean and standard deviation
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # Update running stats
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * std

    # Normalize
    centered_x = x - mean
    normal = centered_x / (std + eps)

    # Apply scale and shift
    out = normal * gamma + beta

    cache = (x, bn_param, gamma, None, None, centered_x, None, None, None, std, normal)

  ## Testing
  elif mode == 'test':
    # Normalize
    out = (x - running_mean) / (running_var + eps)

    # Apply scale and shift
    out *= gamma
    out += beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # Implement the backward pass for batch normalization. Store the            #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  
  # Unpack the cache (cx is centered_x, for brevity)
  _, bn_param, gamma, _, _, cx, _, _, _, std, normal = cache
  eps = bn_param.get('eps', 1e-5)

  # Gradients on beta and gamma
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(normal * dout, axis=0)

  # Gradients on centered inputs
  dcx = dout * gamma / (std + eps) 

  # Compute gradient on inputs X through a simplified expression
  # For additional explanation see: http://cthorey.github.io./backpropagation/
  dx = dcx - np.mean(dcx, axis=0) - cx * np.mean(dcx * cx / (std**2 + eps), axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # Implement the training phase forward pass for inverted dropout.         #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    
    # Each neuron is dropped with probability p and kept with probability 1-p
    mask = (np.random.rand(*x.shape) > p) / (1-p)
    out = x * mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # Implement the test phase forward pass for inverted dropout.             #
    ###########################################################################
    
    out = x

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # Implement the training phase backward pass for inverted dropout.        #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # Implement the convolutional forward pass.                                 #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  stride, pad = conv_param['stride'], conv_param['pad']

  N, _, H, W = x.shape
  F, _, HH, WW = w.shape

  # Initialize output
  H_ = 1 + (H + 2 * pad - HH) / stride
  W_ = 1 + (W + 2 * pad - WW) / stride

  out = np.zeros((N, F, H_, W_))

  # Reshape weights
  w_ = w.reshape(F, -1).T  # (D, F)

  # Pad input along width and height dimensions
  x = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

  # Double loop over every pixel coordinate
  for i in xrange(H_):
    for j in xrange(W_):
      # Slice position
      h_start, h_end = i * stride, i * stride + HH
      w_start, w_end = j * stride, j * stride + WW

      # Input slice
      x_slice = x[:, :, h_start:h_end, w_start:w_end].reshape(N, -1)  # (N, D)

      # Perform affine transform on slice
      out[:, :, i, j] = x_slice.dot(w_) + b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # Implement the convolutional backward pass.                                #
  #############################################################################
  
  # Unpack the cache
  x, w, b, conv_param = cache

  stride, pad = conv_param['stride'], conv_param['pad']
  N, F, H_, W_ = dout.shape
  _, C, HH, WW = w.shape

  # Initialize outputs
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # Double loop over every pixel coordinate
  for i in xrange(H_):
    for j in xrange(W_):
      # Gradient slice (N, F)
      dout_slice = dout[:, :, i, j]

      # Input slice (N, D)
      h_start, h_end = i * stride, i * stride + HH
      w_start, w_end = j * stride, j * stride + WW
      x_slice = x[:, :, h_start:h_end, w_start:w_end].reshape(N, -1)

      # Gradient on bias (F,)
      db += np.sum(dout_slice, axis=0)

      # Gradient on weights (F, D) <--> (F, C, HH, WW)
      dw += dout_slice.T.dot(x_slice).reshape(*dw.shape)

      # Gradient on input slice (N, D) <--> (N, C, HH, WW)
      dx[:, :, h_start:h_end, w_start:w_end] += dout_slice.dot(w.reshape(F, -1)).reshape(N, C, HH, WW)

  # Remove padding
  dx = dx[:, :, pad:-pad, pad:-pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # Implement the max pooling forward pass                                    #
  #############################################################################
  
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H, W = x.shape

  # Number of pools per column/row (output layer width & height)
  H_ = 1 + (H - pool_height) / stride
  W_ = 1 + (W - pool_width) / stride

  # Initialize output
  out = np.zeros((N, C, H_, W_))

  # Double loop over each pixel coordinate/pool
  for i in xrange(H_):
    for j in xrange(W_):
      # Pool position
      h_start, h_end = i * stride, i * stride + pool_height
      w_start, w_end = j * stride, j * stride + pool_width

      # Pool
      pool = x[:, :, h_start:h_end, w_start:w_end]

      # Store max values across width and height
      out[:, :, i, j] = np.max(pool, axis=(2, 3))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # Implement the max pooling backward pass                                   #
  #############################################################################
  
  # Unpack the cache
  x, pool_param = cache

  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  N, C, H_, W_ = dout.shape

  # Initialize output
  dx = np.zeros_like(x)

  # Double loop over each pixel coordinate/pool
  for i in xrange(H_):
    for j in xrange(W_):
      # Pool position
      h_start, h_end = i * stride, i * stride + pool_height
      w_start, w_end = j * stride, j * stride + pool_width

      # Pool
      pool = x[:, :, h_start:h_end, w_start:w_end]

      # Get max pool values
      pool_max = np.max(pool, axis=(2, 3))

      # Brief explanation: gradient only flows through pool maximums. To implement this,
      # I resize (broadcast) gradient to pool size and then apply a mask (also broadcast) 
      # to it, in order to set it to 0 at non-maximum pool values. It would be a lot easier 
      # to do another double loop over dimensions N and C, however this is slightly faster.
    
      # Broadcast them to pool shape to create a mask
      pool_max = np.broadcast_to(pool_max.reshape(N, C, 1), (N, C, pool_height * pool_width))
      pool_max = pool_max.reshape(*pool.shape)
      
      mask = pool == pool_max

      # Broadcast gradient to pool shape
      dpool = np.broadcast_to(dout[:, :, i, j].reshape(N, C, 1), (N, C, pool_height * pool_width))
      dpool = dpool.reshape(*pool.shape)

      # Apply masked gradient
      dx[:, :, h_start:h_end, w_start:w_end] += mask * dpool

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # Implement the forward pass for spatial batch normalization.               #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  # Dimensions
  N, C, H, W = x.shape
  
  # Reshape from 4D to 2D
  x_2d = x.transpose((0, 2, 3, 1)).reshape(N * H * W, C)

  # Perform vanilla batchnorm forward pass
  out_2d, cache = batchnorm_forward_alt(x_2d, gamma, beta, bn_param)

  # Reshape result from 2D to 4D
  out = out_2d.reshape(N, H, W, C).transpose((0, 3, 1, 2))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # Implement the backward pass for spatial batch normalization.              #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  
  # Dimensions
  N, C, H, W = dout.shape
  
  # Reshape from 4D to 2D
  dout_2d = dout.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
  
  # Perform vanilla batchnorm backward pass
  dx_2d, dgamma, dbeta = batchnorm_backward_alt(dout_2d, cache)

  # Reshape result from 2D to 4D
  dx = dx_2d.reshape(N, H, W, C).transpose((0, 3, 1, 2))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
