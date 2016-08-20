import numpy as np
from utilities.classifiers.linear_svm import *
from utilities.classifiers.softmax import *

class LinearClassifier(object):

	def __init__(self):
		self.W = None

	def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
		"""
		Train this linear classifer using stochastic gradient descent.

		Inputs:
		- X: a numpy array of shape (N, D) containing training data. There
		  are N training samples each of dimension D.
		- y: a numpy array of shape (N,) containing training labels. y[i] = c
		  means that X[i] has label 0 <= c < C for C classes.
		- learning_rate: (float) learning rate for optimization.
		- reg: (float) regluarization strength.
		- num_iters: (int) number of steps to take when optimizing
		- batch_size: (int) number of training examples to use at each step.
		- verbose: (boolean) if true, print process during optimization.

		Outputs a list containing the value of the loss at each iteration.
		"""

		num_train, dim = X.shape
		num_classes = np.max(y) + 1

		if self.W is None:
			# lazily initialize W
			self.W = 0.001 * np.random.randn(dim, num_classes)

		# run stochastic gradient descent to optimize W
		loss_history = []
		for it in range(num_iters):
			X_batch = None
			y_batch = None

			# ===================================================================== #
			# Sample batch_size elements from the training data and their           #
			# corresponding labels to use in this round of gradient descent.        #
			# Store the data in X_batch and their corresponding labels in           #
			# y_batch; after sampling X_batch should have shape (dim, batch_size)   #
			# and y_batch should have shape (batch_size,)                           #
			#                                                                       #
			# Hint: Use np.random.choice to generate indices. Sampling with         #
			# replacement is faster than sampling without replacement.              #
			# ===================================================================== #

			mask = np.random.choice(num_train, batch_size, replace=True)
			X_batch = X[mask, :]
			y_batch = y[mask]

			# ===================================================================== #

			# evaluate loss and gradient
			loss, grad = self.loss(X_batch, y_batch, reg)
			loss_history.append(loss)

			# ===================================================================== #
			# Update the weights using the gradient and the learning rate.          #
			# ===================================================================== #

			# perform parameter update
			self.W -= learning_rate * grad

			# ===================================================================== #

			if verbose and it % 100 == 0:
				print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

		return loss_history

	def predict(self, X):
		"""
		Use the trained weights of this linear classifier to predict labels for
		data points.

		Inputs:
		- X: (D x N) array of training data. Each column is a D-dimensional point.

		Returns:
		- y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
		array of length N, and each element is an integer giving the predicted
		class.
		"""

		y_pred = np.zeros(X.shape[1])

		# ======================================================================= #
		# Implement this method. Store the predicted labels in y_pred.            #
		# ======================================================================= #

		# grab highest score for each column
		y_pred = np.argmax(X.dot(self.W), axis=1)

		# ======================================================================= #

		return y_pred

	def loss(self, X_batch, y_batch, reg):
		"""
		Compute the loss function and its derivative. 
		Subclasses will override this.

		Inputs:
		- X_batch: A numpy array of shape (N, D) containing a minibatch of N
		data points; each point has dimension D.
		- y_batch: A numpy array of shape (N,) containing labels for the minibatch.
		- reg: (float) regularization strength.

		Returns: A tuple containing:
		- loss as a single float
		- gradient with respect to self.W; an array of the same shape as W
		"""
		pass

class LinearSVM(LinearClassifier):
	""" 
	A subclass that uses the Multiclass SVM loss function 
	"""
	def loss(self, X_batch, y_batch, reg):
		return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
	""" 
	A subclass that uses the Softmax + Cross-entropy loss function 
	"""
	def loss(self, X_batch, y_batch, reg):
		return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)		