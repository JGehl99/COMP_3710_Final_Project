# Classification Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Neural Network Model
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt

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

DecisionTreeClassifier_model = DecisionTreeClassifier(
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

KNN_model = KNeighborsClassifier(
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
MLPClassifier_model = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=100000,
    shuffle=True
)


# Set to true to display plots.
do_plots = False

# Looping over training/testing sets, train model, calculate prediction, print accuracy.
for train_test in train_test_arr:

    # train_test = [x_train, x_test, y_train, y_test]

    test_size = (len(train_test[0]) / num_rows) * 100

    print('\n=============================================================================')
    print('Train: ', format(test_size, '.3f'), '%  Test: ', format(100-test_size, '.3f'), '%')
    print('=============================================================================\n')

    DecisionTreeClassifier_model.fit(train_test[0], train_test[2])
    KNN_model.fit(train_test[0], train_test[2])
    MLPClassifier_model.fit(train_test[0], train_test[2])

    # Create Predictions.
    DecisionTreeClassifier_prediction = DecisionTreeClassifier_model.predict(train_test[1])
    KNN_prediction = KNN_model.predict(train_test[1])
    MLPClassifier_prediction = MLPClassifier_model.predict(train_test[1])

    # Print Accuracy.
    print("Decision Tree Accuracy:         ", accuracy_score(DecisionTreeClassifier_prediction, train_test[3]))
    print("K-Nearest Neighbour Accuracy:   ", accuracy_score(KNN_prediction, train_test[3]))
    print("Multilayer Perceptron Accuracy: ", accuracy_score(MLPClassifier_prediction, train_test[3]))

    # Only create the plots if set to do so.
    if not do_plots:
        continue

    # Plot the first model.
    DecisionTreeClassifier_Display = RocCurveDisplay.from_estimator(DecisionTreeClassifier_model, train_test[1], train_test[3])

    # For some reason, if we don't first make a dummy plot with just the first model, this will appear twice on the
    # later plot. Comment the below line and run to see what I mean. I can't seem to find a fix for this, and the
    # scikit docs example shows it this way as well.
    plt.show()

    # Plot the remaining two models on top of the first model.
    ax = plt.gca()
    KNN_Display = RocCurveDisplay.from_estimator(KNN_model, train_test[1], train_test[3], ax=ax, alpha=0.8)
    MLPClassifier_Display = RocCurveDisplay.from_estimator(MLPClassifier_model, train_test[1], train_test[3], ax=ax, alpha=0.8)
    DecisionTreeClassifier_Display.plot(ax=ax, alpha=0.8)

    # Show the comparison of all models.
    plt.show()

    # Give a moment to save the plot as something relevant as otherwise you will likely generate too much and lose
    # track of what plot goes with what.
    input('Hit enter to continue...')
