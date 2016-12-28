import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, dropout=False):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - dropout: Boolean; if true, use dropout with p=0.5
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size) + 0.1  # slightly positive so that ReLU doesn't kill too many neurons
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size) 

    # EXTRA FEATURE: Dropout
    self.dropout = dropout
    self.p = 0.35
    self.p_decay = 1.0 + 3e-4 + 5e-5

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # Perform the forward pass, computing the class scores for the input.       #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # Compute hidden layer (affine - ReLU)
    H = np.maximum(0, X.dot(W1) + b1)

    # EXTRA FEATURE: Dropout; kill neurons at random
    if self.dropout and not y is None:
      self.p = self.p * self.p_decay
      dropout_mask = (np.random.rand(*H.shape) < self.p) / self.p
      H *= dropout_mask

    # Compute scores (affine)
    scores = H.dot(W2) + b2 

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # Finish the forward pass, and compute the loss. This should include        #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    
    # Calculate softmax probabilities
    scores -= np.max(scores, axis=1, keepdims=True)  # for numerical stability
    scores_exp = np.exp(scores)
    probabilities = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

    # Calculate loss, accounting for N and regularization
    loss = np.sum(-np.log(probabilities[np.arange(N), y])) / N
    loss += 0.5 * reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # Compute the backward pass, computing the derivatives of the weights       #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    dprobabilities = probabilities.copy()
    dprobabilities[np.arange(N), y] -= 1
    dprobabilities /= N  # so we don't have to do this at the end

    # For biases, we just sum up backproped gradient column-wise.
    db2 = np.sum(dprobabilities, axis=0)
    
    # Gradient of factor in multiplication is multiplication of other factor with backproped gradient.
    dW2 = H.T.dot(dprobabilities)
    dH = dprobabilities.dot(W2.T)

    # If dropout is used, we need to account for it in backprop
    if self.dropout:
      dH *= dropout_mask

    # Mustn't forget to backprop on ReLU! Set gradient to zero where H is zero or less.
    dH[H <= 0] = 0

    # Rinse and repeat
    db1 = np.sum(dH, axis=0)
    dW1 = X.T.dot(dH)

    # Store gradients and account for regularization
    grads['W2'] = dW2 + reg * W2
    grads['W1'] = dW1 + reg * W1
    grads['b2'] = db2
    grads['b1'] = db1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # Create a random minibatch of training data and labels, storing        #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      
      N = X.shape[0]

      # Produce a batch by picking random indices
      indices = np.random.choice(N, batch_size)
      X_batch = X[indices, :]
      y_batch = y[indices]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # Use the gradients in the grads dictionary to update the               #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      for parameter in self.params.keys():
        self.params[parameter] -= learning_rate * grads[parameter]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # Implement this function; it should be VERY simple!                      #
    ###########################################################################

    # Get best-scoring class by picking the highest number's index in a row 
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


