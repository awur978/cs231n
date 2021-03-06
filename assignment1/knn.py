# loading packages
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from utilities.data_utils import load_CIFAR10
from utilities.classifiers.k_nearest_neighbor import KNearestNeighbor

def time_function(f, *args):
	"""
	Call a function f with args and return the time (in seconds) that it took to execute.
	"""
	tic = time.time()
	f(*args)
	toc = time.time()

	return toc - tic

# Load the raw CIFAR-10 data.
cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('\nTraining data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
	idxs = np.flatnonzero(y_train == y)
	idxs = np.random.choice(idxs, samples_per_class, replace=False)
	for i, idx in enumerate(idxs):
		plt_idx = i * num_classes + y + 1
		plt.subplot(samples_per_class, num_classes, plt_idx)
		plt.imshow(X_train[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance. 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train) # no-op

dists = classifier.compute_distances_two_loops(X_test)

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got {} / {} correct => accuracy: {}'.format(num_correct, num_test, accuracy))

# Let's compare how fast the implementations are
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took {} seconds'.format(two_loop_time))

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took {} seconds'.format(one_loop_time))

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took {} seconds'.format(no_loop_time))

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: {}'.format(difference, ))
if difference < 0.001:
	print('Good! The distance matrices are the same')
else:
	print('Uh-oh! The distance matrices are different')

# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: {}'.format(difference, ))
if difference < 0.001:
	print('Good! The distance matrices are the same')
else:
	print('Uh-oh! The distance matrices are different')

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

# ============================================================================ #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
# ============================================================================ #

# grab number of training observations
num_train = X_train.shape[0]
# indices => split num_train into 5 folds
range_split = np.array_split(range(num_train), num_folds)

y_train_folds = [y_train[range_split[i]] for i in range(num_folds)]
X_train_folds = [X_train[range_split[i]] for i in range(num_folds)]

# ============================================================================ #
# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

# ============================================================================ #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
# ============================================================================ #

# loop over values of k
for k in k_choices:
	print('k = {}...'.format(k))
	# loop over folds
	for fold in range(num_folds): 
		# use i'th fold as validation set
		X_valid = X_train_folds[fold]
		y_valid = y_train_folds[fold]

		# concatenate before x[fold] and after x[fold]
		temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
		temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])

		# instantiate classifier
		temp_classifier = KNearestNeighbor()
		temp_classifier.train(temp_X_train, temp_y_train)

		# compute distances
		temp_dists = temp_classifier.compute_distances_no_loops(X_valid)
		temp_y_test_preds = temp_classifier.predict_labels(temp_dists, k=k)

		# check accuracies
		num_correct = np.sum(temp_y_test_preds == y_valid)
		num_test = X_valid.shape[0]
		accuracy = float(num_correct) / num_test
		k_to_accuracies[k] = k_to_accuracies.get(k,[]) + [accuracy]

# ============================================================================ #

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = {}, accuracy = {}'.format(k, accuracy))

# plot the raw observations
for k in k_choices:
	accuracies = k_to_accuracies[k]
	plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = k_choices[np.argmax(accuracies_mean)]
print('Best k = {}'.format(best_k))

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got {} / {} correct => accuracy: {}'.format(num_correct, num_test, accuracy))
