# Classification Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

# Neural Network Model
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt

# Set to true to display plots.
do_plots = True

# Loading data from the CSV file.
data = pd.read_csv('mushrooms.csv', header=None)

num_rows = len(data.index)

# Separating features from class label.
features = data.iloc[:, 1:]
class_label = data.iloc[:, 0]

# Split data into training and testing sets.
# x_train, x_test, y_train, y_test

train_test_arr = []

for x in range(1, 10):
    train_test_arr.append(train_test_split(features, class_label, test_size=1-float(format(x * 0.001, '.3f'))))

for x in range(1, 10):
    train_test_arr.append(train_test_split(features, class_label, test_size=1 - float(format(x * 0.01, '.2f'))))

for x in range(1, 10):
    train_test_arr.append(train_test_split(features, class_label, test_size=1-float(format(x*0.1, '.1f'))))

# Decision Tree
#
#   criterion = {“gini”, “entropy”}, default=”gini”
#       The function to measure the quality of a split.
#       Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#
#   splitter = {“best”, “random”}, default=”best”
#       The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and
#       “random” to choose the best random split.
#
#   max_depth - int, default=None
#       The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
#       contain less than min_samples_split samples.
#
#   min_samples_split - int or float, default=2
#       The minimum number of samples required to split an internal node:
#       If int, then consider min_samples_split as the minimum number.
#       If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of
#       samples for each split.
#
#   max_features - int, float or {“auto”, “sqrt”, “log2”}, default=None
#       The number of features to consider when looking for the best split:
#       If int, then consider max_features features at each split.
#       If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
#       If “auto”, then max_features=sqrt(n_features).
#       If “sqrt”, then max_features=sqrt(n_features).
#       If “log2”, then max_features=log2(n_features).
#       If None, then max_features=n_features.

model_DecisionTreeClassifier = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    max_features=None
)

# K-Nearest Neighbour
#
#   n_neighbors - int, default=5
#       Number of neighbors to use by default for k-neighbors queries.
#
#   weights = {‘uniform’, ‘distance’} or callable, default=’uniform’
#       Weight function used in prediction. Possible values:
#       ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#       ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point
#       will have a greater influence than neighbors which are further away.
#       [callable] : a user-defined function which accepts an array of distances, and returns an array of the same
#       shape containing the weights.
#
#   algorithm = {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
#       Algorithm used to compute the nearest neighbors:
#           ‘ball_tree’ will use BallTree
#           ‘kd_tree’ will use KDTree
#           ‘brute’ will use a brute-force search.
#           ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
#       Note: fitting on sparse input will override the setting of this parameter, using brute force.
#
#   leaf_size - int, default=30
#       Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the
#       memory required to store the tree. The optimal value depends on the nature of the problem.

model_KNeighborsClassifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30
)


# Multilayer Perceptron Model
#
#   hidden_layer_sizes - tuple, length = n_layers-2, default=(100,)
#       The ith element represents the number of neurons in the ith hidden layer.
#
#   activation = {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
#       Activation function for the hidden layer.
#       ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
#       ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
#       ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
#       ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
#
#   solver = {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
#       The solver for weight optimization.
#       ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
#       ‘sgd’ refers to stochastic gradient descent.
#       ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
#
#   learning_rate = {‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
#       Learning rate schedule for weight updates.
#       ‘constant’ is a constant learning rate given by ‘learning_rate_init’.
#       ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of
#       ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
#       ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing.
#       Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation
#       score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
#       Only used when solver='sgd'.
#
#   learning_rate_init - float, default=0.001
#       The initial learning rate used. It controls the step-size in updating the weights.
#       Only used when solver=’sgd’ or ‘adam’.
#
#   power_t - float, default=0.5
#       The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the
#       learning_rate is set to ‘invscaling’.
#       Only used when solver=’sgd’.
#
#   max_iter - int, default=200
#       Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of
#       iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
#       (how many times each data point will be used), not the number of gradient steps.
#
#   shuffle - bool, default=True
#       Whether to shuffle samples in each iteration.
#       Only used when solver=’sgd’ or ‘adam’.
#
model_MLPClassifier = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=100000,
    shuffle=True
)


# Linear Regressions Model
#
#   fit_intercept - bool, default=True
#       Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
#       (i.e. data is expected to be centered).
#
#   copy_X - bool, default=True
#       If True, X will be copied; else, it may be overwritten.
#
#   n_jobs - int, default=None
#       The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large
#       problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1
#       unless in a joblib.parallel_backend context. -1 means using all processors.
#
model_LinearRegression = LinearRegression(
    fit_intercept=True,
    copy_X=True,
    n_jobs=None
)


# Stochastic Gradient Descent Regressor
#
#   loss - str, default=’squared_error’
#       The loss function to be used. The possible values are:
#       ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
#
#       The ‘squared_error’ refers to the ordinary least squares fit. ‘huber’ modifies ‘squared_error’ to focus less on
#       getting outliers correct by switching from squared to linear loss past a distance of epsilon.
#       ‘epsilon_insensitive’ ignores errors less than epsilon and is linear past that; this is the loss function used
#       in SVR. ‘squared_epsilon_insensitive’ is the same but becomes squared loss past a tolerance of epsilon.
#
#   penalty = {‘l2’, ‘l1’, ‘elasticnet’}, default=’l2’
#       The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear
#       SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’
#
#   alpha - float, default=0.0001
#       Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also
#       used to compute the learning rate when set to learning_rate is set to ‘optimal’.
#
#   l1_ratio - float, default=0.15
#       The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
#       Only used if penalty is ‘elasticnet’.
#
#   fit_intercept - bool, default=True
#       Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
#
#   max_iter - int, default=1000
#       The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method,
#       and not the partial_fit method.
#
#   tol - float, default=1e-3
#       The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change
#       consecutive epochs. Convergence is checked against the training loss or the validation loss depending on the
#       early_stopping parameter.
#
#   shuffle - bool, default=True
#       Whether or not the training data should be shuffled after each epoch.
#
#   verbose - int, default=0
#       The verbosity level.
#
#   epsilon - float, default=0.1
#       Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or
#       ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the
#       prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct
#       label are ignored if they are less than this threshold.
#
#
#   learning_rate - str, default=’invscaling’
#       The learning rate schedule:
#
#       ‘constant’: eta = eta0
#       ‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
#       ‘invscaling’: eta = eta0 / pow(t, power_t)
#
#       ‘adaptive’: eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs
#       fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is
#       True, the current learning rate is divided by 5.
#
#   eta0 - float, default=0.01
#       The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.01.
#
#   power_t - float, default=0.25
#       The exponent for inverse scaling learning rate.
#

model_SGDRegressor = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    learning_rate='invscaling',
    eta0=0.01,
    power_t=0.25
)


# Looping over training/testing sets, train model, calculate prediction, print accuracy.
for train_test in train_test_arr:

    # train_test = [x_train, x_test, y_train, y_test]

    test_size = (len(train_test[0]) / num_rows) * 100

    print('\n=============================================================================')
    print('Train: ', format(test_size, '.3f'), '%  Test: ', format(100-test_size, '.3f'), '%')
    print('=============================================================================\n')

    # Fit models to training data.
    model_DecisionTreeClassifier.fit(train_test[0], train_test[2])
    model_KNeighborsClassifier.fit(train_test[0], train_test[2])
    model_MLPClassifier.fit(train_test[0], train_test[2])
    model_LinearRegression.fit(train_test[0], train_test[2])
    model_SGDRegressor.fit(train_test[0], train_test[2])

    # Test models against test data.
    print("Decision Tree Accuracy:         ", model_DecisionTreeClassifier.score(train_test[1], train_test[3]))
    print("K-Nearest Neighbour Accuracy:   ", model_KNeighborsClassifier.score(train_test[1], train_test[3]))
    print("Multilayer Perceptron Accuracy: ", model_MLPClassifier.score(train_test[1], train_test[3]))
    print("Linear Regression Accuracy:     ", model_LinearRegression.score(train_test[1], train_test[3]))
    print("SGD Regression Accuracy:        ", model_SGDRegressor.score(train_test[1], train_test[3]))

    # Only create the plots if set to do so.
    if not do_plots:
        continue

    # Plot the first model.
    display_DecisionTreeClassifier = RocCurveDisplay.from_estimator(model_DecisionTreeClassifier, train_test[1], train_test[3])

    # For some reason, if we don't first make a dummy plot with just the first model, this will appear twice on the
    # later plot. Comment the below line and run to see what I mean. I can't seem to find a fix for this, and the
    # scikit docs example shows it this way as well.
    plt.show()

    # Plot the remaining models on top of the first model.
    ax = plt.gca()
    display_KNeighborsClassifier = RocCurveDisplay.from_estimator(model_KNeighborsClassifier, train_test[1], train_test[3], ax=ax, alpha=0.8)
    display_MLPClassifier = RocCurveDisplay.from_estimator(model_MLPClassifier, train_test[1], train_test[3], ax=ax, alpha=0.8)
    display_DecisionTreeClassifier.plot(ax=ax, alpha=0.8)

    # Show the comparison of all models.
    plt.show()

    # Give a moment to save the plot as something relevant as otherwise you will likely generate too much and lose
    # track of what plot goes with what.
    input('Hit enter to continue...')
