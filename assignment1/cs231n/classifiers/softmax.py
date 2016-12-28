import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Number of training images and classes, respectively
  N = X.shape[0]
  C = W.shape[1]

  # Intermediate matrix for gradient computation
  dprobabilities = np.zeros((N, C))

  for i in xrange(N):
    # Compute class scores
    scores = X[i].dot(W)

    # Offset them by max, for numerical stability
    scores -= np.max(scores)

    # Compute probabilities
    probabilities = np.exp(scores) / np.sum(np.exp(scores))

    # Add correct class negative log probability to loss
    loss += -np.log(probabilities[y[i]])

    # Add probabilities to intermediate matrix
    dprobabilities[i, :] = probabilities
    dprobabilities[i, y[i]] -= 1

  # Average loss, account for regularization
  loss /= N
  loss += 0.5 * reg * np.sum(np.square(W))

  # Compute gradient from intermediate matrix
  dW = X.T.dot(dprobabilities)
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = X.shape[0]

  # Compute scores and probabilities using vector functions
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)  # for numerical stability
  scores_exp = np.exp(scores)
  probabilities = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

  # Not much is changed, we just drop the i index
  loss += np.sum(-np.log(probabilities[np.arange(N), y])) / N
  loss += 0.5 * reg * np.sum(np.square(W))

  # Compute gradient, very similar to above
  dprobabilities = probabilities
  dprobabilities[np.arange(N), y] -= 1

  dW = X.T.dot(dprobabilities)
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

