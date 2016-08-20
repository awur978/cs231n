import numpy as np

class KNearestNeighbor(object):
	"""
	A kNN classifier with L2 distance.
	"""

	def __init__(self):
		pass

	def train(self, X, y):
		"""
		Train the classifer. This is equivalent to just
		memorizing the training data.

		Inputs:
		- X: a numpy array of shape (num_train, D) containing the
		  the training data consisting of num_train samples each of
		  dimension D.
		- y: A numpy array of shape (N, ) containing the training
		  labels, where y[i] is the label for X[i].
		"""

		self.X_train = X
		self.y_train = y

	def predict(self, X, k=1, num_loops=0):
		"""
		Predict labels for test data using this classifier.

		Inputs:
		- X: a numpy array of shape (num_test, D) containing test data
		  consisting of num_test samples each of dimension D/
		- k: the number of nearest neighbors that vote for the predicted
		  labels.
		- num_loops: determines which implementation to use to compute
		  distances between training points and testing points.
		  0 corresponds to a fully vectorized implementation while 2 uses
		  2 for loops.

		Returns:
		- y: a numpy array of shape (num_test, ) containing predicted labels
		  for the test data, where y[i] is the predicted label for the test
		  point X[i].
		"""

		if num_loops == 0:
			dists = self.compute_distances_no_loops(X)
		elif num_loops == 1:
			dists = self.compute_distances_one_loop(X)
		elif num_loops == 2:
			dists = self.compute_distances_two_loops(X)
		else:
			raise ValueError('Invalid value {} for num_loops'.format(num_loops))

		return self.predict_labels(dists, k=k)

	def compute_distances_two_loops(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using a nested loop over both the training data and the 
		test data.

		Inputs:
		- X: a numpy array of shape (num_test, D) containing test data.

		Returns:
		- dists: a numpy array of shape (num_test, num_train) where dists[i, j]
		is the Euclidean distance between the ith test point and the jth training
		point.
	    """

		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))

		# ================================================================= #
		# Compute the L2 distance between the ith test point and the jth    #
		# training point, and store the result in dists[i, j]. You should   #
		# not use a loop over dimension.                                    #
		# ================================================================= #

		for i in range(num_test):
			for j in range(num_train):
				# compute square difference
				sq_diff = (X[i, :] - self.X_train[j, :])**2
				# sum over D dimensions
				summation = np.sum(sq_diff)
				# sqrt the whole and assign
				dists[i, j] = np.sqrt(summation)

		# ================================================================= #

		return dists

	def compute_distances_one_loop(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using a single loop over the test data.

		Input / Output: Same as compute_distances_two_loops
		"""

		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))

		# ==================================================================== #
		# Compute the l2 distance between the ith test point and all training  #
		# points, and store the result in dists[i, :].                         #
		# ==================================================================== #

		for i in range(num_test):
			# compute square difference
			sq_diff = np.square(self.X_train - X[i, :])
			# compute sum
			summation = np.sum(sq_diff, axis=1)
			# square root
			dists[i, :] = np.sqrt(summation)

		# ==================================================================== #

		return dists

	def compute_distances_no_loops(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using no explicit loops.

		Input / Output: Same as compute_distances_two_loops
		"""

		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train)) 

		# ===================================================================== #
		# Compute the l2 distance between all test points and all training      #
		# points without using any explicit loops, and store the result in      #
		# dists.                                                                #
		#                                                                       #
		# You should implement this function using only basic array operations; #
		# in particular you should not use functions from scipy.                #
		#                                                                       #
		# HINT: Try to formulate the l2 distance using matrix multiplication    #
		#       and two broadcast sums.                                         #
		# ===================================================================== #

		sum1 = np.sum(np.square(X), axis=1)
		sum2 = np.sum(np.square(self.X_train), axis=1)
		dot_product = np.dot(X, self.X_train.T)

		dists = np.sqrt(sum1[:, np.newaxis] + sum2 - 2 * dot_product)

		# ===================================================================== #

		return dists

	def predict_labels(self, dists, k=1):
		"""
		Given a matrix of distances between test points and training points,
		predict a label for each test point.

		Inputs:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		gives the distance betwen the ith test point and the jth training point.

		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
		test data, where y[i] is the predicted label for the test point X[i].  
		"""

		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)

		for i in range(num_test):
			# A list of length k storing the labels of the k nearest neighbors to
			# the ith test point.
			closest_y = []

			# ===================================================================== #
			# Use the distance matrix to find the k nearest neighbors of the ith    #
			# testing point, and use self.y_train to find the labels of these       #
			# neighbors. Store these labels in closest_y.                           #
			# ===================================================================== #

			closest_idx = np.argsort(dists[i, :])[:k].tolist()
			closest_y = self.y_train[closest_idx]

			# ===================================================================== #
			# Now that you have found the labels of the k nearest neighbors, you    #
			# need to find the most common label in the list closest_y of labels.   #
			# Store this label in y_pred[i]. Break ties by choosing the smaller     #
			# label.                                                                #
			# ===================================================================== #

			# count the frequency of the closest labels
			counts = np.bincount(closest_y)
			# return the most frequent item
			y_pred[i] = np.argmax(counts)

			# ===================================================================== #

		return y_pred
