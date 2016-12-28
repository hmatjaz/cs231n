import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # Compute loss and gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_positive_class_margins = 0  # Keep track of number of positive margins
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        num_positive_class_margins += 1  # Add to count
        dW[:, j] += X[i]                 # Gradient here is image i
    
    # Gradient on correct class is negative image i multiplied by
    # number of positive incorrect class margins.
    dW[:, y[i]] += -1.0 * num_positive_class_margins * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Average the loss gradient and add regularization gradient
  dW /= num_train
  dW += reg * W

  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # Dimensions
  N = X.shape[0]

  # Compute scores and margins
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(N), y]

  # reshape() is needed to convert (D,) to (D, 1) so that broadcasting works.
  margins = np.maximum(0, scores - correct_class_scores.reshape(-1, 1) + 1)
  margins[np.arange(N), y] = 0

  # Loss is simply a total of all margins divided by number of images.
  loss = np.sum(margins) / N

  # We also need to account for regularization.
  loss += 0.5 * reg * np.sum(np.square(W))

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # For gradient, we only care whether gradient was either positive or zero
  intermediate = (margins > 0) * 1.0  # * 1.0 converts boolean to float

  # Number of positive class margins per image
  intermediate[np.arange(N), y] = -1.0 * np.sum(intermediate, axis=1)

  # Compute gradient, divide by N and account for regularization
  dW = (X.T.dot(intermediate) / N) + reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
