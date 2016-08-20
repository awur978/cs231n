# loading packages
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from utilities.classifiers.linear_classifier import LinearSVM
from utilities.data_utils import load_CIFAR10
from utilities.gradient_check import grad_check_sparse
from utilities.classifiers.linear_svm import svm_loss_naive
from utilities.classifiers.linear_svm import svm_loss_vectorized

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
	"""
	Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
	it for the linear classifier.  
	"""

	# Load the raw CIFAR-10 data
	cifar10_dir = 'datasets/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	# subsample the data
	mask = range(num_training, num_training + num_validation)
	X_val = X_train[mask]
	y_val = y_train[mask]
	mask = range(num_training)
	X_train = X_train[mask]
	y_train = y_train[mask]
	mask = range(num_test)
	X_test = X_test[mask]
	y_test = y_test[mask]
	mask = np.random.choice(num_training, num_dev, replace=False)
	X_dev = X_train[mask]
	y_dev = y_train[mask]

	# Preprocessing: reshape the image data into rows
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_val = np.reshape(X_val, (X_val.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
	X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

	# Normalize the data: subtract the mean image
	mean_image = np.mean(X_train, axis = 0)
	X_train -= mean_image
	X_val -= mean_image
	X_test -= mean_image
	X_dev -= mean_image

	# add bias dimension (i.e. bias trick) and transform into columns
	X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
	X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

	return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

# load the data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=False)
toc = time.time()
print('That took {}s'.format(toc - tic))

# A useful debugging strategy is to plot the loss as a function of iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print('training accuracy: {}'.format(np.mean(y_train == y_train_pred), ))
y_val_pred = svm.predict(X_val)
print('validation accuracy: {}'.format(np.mean(y_val == y_val_pred), ))

# ============================================================================ #
# Use the validation set to tune hyperparameters (regularization strength and  #
# learning rate). You should experiment with different ranges for the learning #
# rates and regularization strengths; if you are careful you should be able to #
# get a classification accuracy of about 0.4 on the validation set.            #
# Results is dictionary mapping tuples of the form 							   #
# (learning_rate, regularization_strength) to tuples of the form               #
# (training_accuracy, validation_accuracy). The accuracy is simply the         #
# fraction of data points that are correctly classified.                       #
# ============================================================================ #

learning_rates = [3e-7, 5e-8,9e-8]
regularization_strengths = [3e5, 5e5]
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

# ============================================================================ #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
# ============================================================================ #

for lr in learning_rates:
	for rs in regularization_strengths:
		# instantiate classifier
		svm = LinearSVM()

		# train classifier with given lr and rs
		svm.train(X_train, y_train, learning_rate=lr, reg=rs, 
				  num_iters=1500, verbose=True)

		# predict values for train and valid
		y_train_pred = svm.predict(X_train)
		y_val_pred = svm.predict(X_val)

		# compute accuracies
		train_accuracy = np.mean(y_train == y_train_pred)
		val_accuracy = np.mean(y_val == y_val_pred)

		if val_accuracy > best_val:
			best_val = val_accuracy
			best_svm = svm

		# store accuracies for given lr and rs
		results[(lr, rs)] = train_accuracy, val_accuracy
# ============================================================================ #

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr {} reg {} train accuracy: {} val accuracy: {}'.format(
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: {}'.format(best_val))


# ============================================================================ #

# Visualize the cross-validation results
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()

# ============================================================================ #

# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: {}'.format(test_accuracy))

# ============================================================================ #

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.

w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(10):
	plt.subplot(2, 5, i + 1)
    
	# Rescale the weights to be between 0 and 255
	wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
	plt.imshow(wimg.astype('uint8'))
	plt.axis('off')
	plt.title(classes[i])
plt.show()
