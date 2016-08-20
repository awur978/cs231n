# loading packages
import random
import numpy as np
import matplotlib.pyplot as plt

from utilities.data_utils import load_CIFAR10
from utilities.features import *
from utilities.classifiers.neural_net import TwoLayerNet

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the train and test sets from the cifar-10 directory and create subsample
    sets from them.
    
    Inputs:
    - num_training: size of train set to be taken from original training set.
    - num_validation: size of validation set taken from original training set.
    - num_test: size of test set to be taken from original test set.

    Output:
    - X_train, y_train, X_val, y_val, X_test, y_test. 
    """
   
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test

def feature_extraction(train, val, test, feature_fns):
    """
    Perform feature extraction of the train, validation and test sets.

    Input:
    - train: a numpy array containing the training samples.
    - val: a numpy array containing the validation samples.
    - test: a numpy array containing the test samples.
    - feature_fns: list of k feature functions.
    """

    # ectract features
    X_train_feats = extract_features(X_train, feature_fns, verbose=True)
    X_val_feats = extract_features(X_val, feature_fns)
    X_test_feats = extract_features(X_test, feature_fns)

    # preprocessing: subtract the mean feature
    mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # preprocessing: divide by standard deviation. This ensures that each feature
    # has roughly the same scale.
    std_feat = np.std(X_train_feats, axis=0, keepdims=True)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # preprocessing: add a bias dimension
    X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
    X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
    X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

    return X_train_feats, X_val_feats, X_test_feats

def run_cross_validation(learning_rates, regularization_strengths, learning_rate_decays, verbose=False):
    """
    Tune hyperparameters using the validation set.

    Inputs:
    - learning_rates: list of different values for lr
    - hidden_sizes: list of different values for hs
    - learning_rate_decays: list of different values for lrd
    - regularization_strengths= list of different values for rs

    Output:
    - dict returning train and valid accuracy for given lr, rs, hs and lrd.  
    """

    results = {}
    best_net = None
    best_val = -1 # highest validation accuracy that we have seen so far.

    for lr in learning_rates:
        for rs in regularization_strengths:
            for lrd in learning_rate_decays:

                # print params
                print('\nTraining params: learning rate: {}, reg strength: {}, lr decay: {}'.format(lr, rs, lrd))
                
                # instantiate NN
                net = TwoLayerNet(input_size, hidden_size, num_classes)

                # train it
                stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
                                  num_iters=2000, batch_size=200, learning_rate=lr, 
                                  learning_rate_decay=lrd, reg=rs, verbose=True)

                # plot the loss history
                if verbose:
                    plt.plot(stats['loss_history'])
                    plt.xlabel('iteration')
                    plt.ylabel('training loss')
                    plt.title('Training Loss history')
                    plt.show()

                # make predictions
                y_train_pred = net.predict(X_train_feats)
                y_val_pred = net.predict(X_val_feats)

                # compute accuracy
                train_accuracy = np.mean(y_train == y_train_pred)
                val_accuracy = np.mean(y_val == y_val_pred)

                # save if better
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_net = net

                # store in dictionary
                results[(lr, rs, lrd)] = (train_accuracy, val_accuracy)

    return results, best_val

# load the data
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# number of bins in the color histogram
num_color_bins = 10
# define feature functions (HOG and color histogram)
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]

# extract the features
X_train_feats, X_val_feats, X_test_feats = feature_extraction(X_train, X_val, X_test, feature_fns=feature_fns)

# neural network params
input_size = X_train_feats.shape[1]
hidden_size = 500
num_classes = 10

# tuning params to try out
learning_rates = [0.85, 0.80]
regularization_strengths = [0.009, 0.003]
learning_rate_decays = [0.9]

# hyperparameter tuning
results, best_val = run_cross_validation(learning_rates, regularization_strengths, learning_rate_decays)

print('\n')
for lr, reg, lrd in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, lrd)]
    print('lr %e reg %e lrd %e train accuracy: %f val accuracy: %f' % (
                lr, reg, lrd, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)

# get best params
lr, rs, lrd = max(results, key=lambda x: results[x[0:3]])

net = TwoLayerNet(input_size, hidden_size, num_classes)
print('\nTraining Fully-Connected 2-Layer Neural Network with best parameters...')
stats = net.train(X_train_feats, y_train, X_val_feats, y_val, num_iters=3000, batch_size=200, learning_rate=lr, 
                  learning_rate_decay=lrd, reg=rs, verbose=True)

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

y_test_pred = net.predict(X_test_feats)

test_acc = (y_test_pred == y_test).mean()
print('Test accuracy: {:f}'.format(test_acc))

# view misclassified examples
examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()
