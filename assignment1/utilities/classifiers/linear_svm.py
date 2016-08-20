import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

	Inputs have dimension D, there are C classes, and we operate on 
	minibatches of N examples.

	Inputs:
	- W: a numpy array of shape (D, C) containing weights.
	- X: a numpy array of shape (N, D) containing a minibatch of data.
	- y: a numpy array of shape (N,) contaning training labels. y[i] = c
	  means that X[i] has label c, where 0 <= c <= C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float.
	- gradient with respect to weight W; an array of same shape as W.
	"""

	delta = 1.0
	loss = 0.0
	dW = np.zeros(W.shape)
	num_classes = W.shape[1]
	num_train = X.shape[0]

	# loop over each image
	for i in range(num_train):
		scores = np.dot(X[i], W)
		# grab correct class score
		correct_class_score = scores[y[i]]
		# iterate over wrong classes
		for j in range(num_classes):
			if j == y[i]:
				# ignore correct class
				continue
			# only compute loss if incorrectly classified
			margin = scores[j] - correct_class_score + delta
			if margin > 0:
				# accumulate loss
				loss += margin
				# correct class gradient
				dW[:, y[i]] -= X[i, :]
				# incorrect classes gradient
				dW[:, j] += X[i, :]


	# average the loss and gradient
	loss /= num_train
	dW /= num_train

	# add regularization contribution
	loss += 0.5 * reg * np.sum(W*W)
	dW += reg * W

	return loss, dW

def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	delta = 1.0
	dW = np.zeros(W.shape)
	num_train = X.shape[0]

	# ========================================================================= #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	# ========================================================================= #

	# compute scores
	scores = np.dot(X, W) # (N x C)

	# grab correct class score
	correct_class_score = scores[np.arange(num_train), y]

	# compute margins
	margins = np.maximum(0, scores.T - correct_class_score + delta)

	# subtract correct class scores from loss (a total of N)
	loss = (np.sum(margins) - num_train) / num_train

	# add reg contribution
	loss += 0.5 * reg * np.sum(W*W)

	# ========================================================================= #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	# ========================================================================= #

	# make entries 1 if > 0, and 0 otherwise
	slopes = np.zeros((margins.shape))
	slopes[margins>0] = 1

	# set elements of row corresponding to correct class to negative of sum
	slopes[y, range(num_train)] -= np.sum(margins>0, axis=0)
	
	# dot product with X and average out
	dW = np.dot(X.T, slopes.T) / float(num_train)

	# add reg contribution
	dW += reg * W 
 
	# ========================================================================= #

	return loss, dW
