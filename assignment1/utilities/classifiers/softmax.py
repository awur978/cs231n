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
	- loss: (float)
	- gradient with respect to weights W; an array of same shape as W
	"""

	num_train = X.shape[0]
	num_classes = W.shape[1]
	loss = 0.0
	dW = np.zeros_like(W) # initialize gradient to 0

	# ========================================================================= #
	# Compute the softmax loss and its gradient using explicit loops.           #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	# ========================================================================= #

	for i in range(num_train):
		# compute unnormlaized log probs
		unorm_log_probs = np.dot(X[i], W)

		# for numerical stability 
		unorm_log_probs -= np.max(unorm_log_probs)

		# get class probabilities
		probs = np.exp(unorm_log_probs) / np.sum(np.exp(unorm_log_probs))

		# compute loss
		loss -= np.log(probs[y[i]])

		# subtract 1 from correct class of prob
		probs[y[i]] -= 1

		for j in range(num_classes):
			dW[:, j] += X[i, :] * probs[j]

	# average out grad and loss
	loss /= num_train
	dW /= num_train

	# add regularization contribution
	loss += 0.5 * reg * np.sum(W * W)
	dW += reg * W

	# ========================================================================= #

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.

	num_train = X.shape[0]
	num_classes = W.shape[1]
	dW = np.zeros_like(W) # initiliaze gradient to 0

	# ========================================================================= #
	# Compute the softmax loss and its gradient using no explicit loops.        #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	# ========================================================================= #

	# compute log probabilities
	unorm_log_probs = np.dot(X, W)
	unorm_log_probs -= np.max(unorm_log_probs, axis=1, keepdims=True)
	unorm_probs = np.exp(unorm_log_probs)
	probs = unorm_probs / np.sum(unorm_probs, axis=1, keepdims=True)

	# compute data and reg loss
	corect_logprobs = -np.log(probs[np.arange(num_train), y])
	data_loss = np.sum(corect_logprobs) / num_train
	reg_loss = 0.5 * reg * np.sum(W * W)
	loss = data_loss + reg_loss

	# compute gradient dW
	probs[np.arange(num_train), y] -= 1
	probs /= num_train
	dW = np.dot(X.T, probs)

	return loss, dW

