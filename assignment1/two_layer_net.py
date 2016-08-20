# loading packages
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from utilities.classifiers.neural_net import TwoLayerNet
from utilities.vis_utils import visualize_grid
from utilities.gradient_check import eval_numerical_gradient
from utilities.data_utils import load_CIFAR10

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
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

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

def run_cv(hidden_sizes, learning_rates, learning_rate_decays, regularization_strengths):
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
    best_val = -1
    best_net = None

    for lr in learning_rates:
        for rs in regularization_strengths:
            for hs in hidden_sizes:
                for lrd in learning_rate_decays:
                    # print params
                    print('Training params: learning rate: {}, reg strength: {}, hidden size: {}, lr decay: {}'.format(lr, rs, hs, lrd))

                    # instantiate NN
                    net = TwoLayerNet(input_size, hs, num_classes)

                    # train it
                    stats = net.train(X_train, y_train, X_val, y_val,
                                      num_iters=1500, batch_size=200,
                                      learning_rate=lr, learning_rate_decay=lrd,
                                      reg=rs, verbose=True)

                    # predict train and valid
                    y_train_pred = net.predict(X_train)
                    y_val_pred = net.predict(X_val)

                    # compute accuracies
                    train_accuracy = np.mean(y_train == y_train_pred)
                    val_accuracy = np.mean(y_val == y_val_pred)

                    # save best network
                    if val_accuracy > best_val:
                        best_val = val_accuracy
                        best_net = net

                    # store best params
                    results[(lr, rs, hs, lrd)] = (train_accuracy, val_accuracy)

    return results, best_val

# load the data
print('Loading the data...')
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

# define NN parameters
input_size = 32 * 32 * 3
num_classes = 10

# adjust these for tuning
hidden_sizes = [50, 100]
learning_rates = [1e-3]
learning_rate_decays = [0.99]
regularization_strengths = [0.8]

# hyperparameter tuning
results, best_val  = run_cross_validation(hidden_sizes, learning_rates, learning_rate_decays, regularization_strengths)

for lr, reg, hs,lrd in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, hs, lrd)]
    print('lr %e reg %e hs %d lrd %e train accuracy: %f val accuracy: %f' % (lr, reg, hs, lrd, train_accuracy, val_accuracy))
    
print('Best validation accuracy achieved during cross-validation: %f' % best_val)

# hypertuning gives the following best parameters
lr, rs, hs, lrd = max(results, key=lambda x: results[x[0:4]])

# instantiate best NN
best_net = TwoLayerNet(input_size, hs, num_classes)
# train it
best_net.train(X_train, y_train, X_val, y_val, num_iters=2000, batch_size=200, 
               learning_rate=lr, learning_rate_decay=lrd, reg=rs, verbose=True)

# visualize the weights of the best network
show_net_weights(best_net)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
