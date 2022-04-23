# Classification Models
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Neural Network Model
from sklearn.neural_network import MLPClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Set to true to generate and display plots.
do_plots = False

# Loading data from csv file
data = pd.read_csv('mushrooms.csv', header=None)

num_rows = len(data.index)

# Separating features from class label
features = data.iloc[:, 1:]
class_label = data.iloc[:, 0]

# Arrays to hold training sets
test_train_1 = []
test_train_2 = []
test_train_3 = []
test_train_4 = []
test_train_5 = []
test_train_6 = []
test_train_7 = []
test_train_8 = []
test_train_9 = []
test_train_10 = []
test_train_20 = []
test_train_30 = []
test_train_40 = []
test_train_50 = []
test_train_60 = []
test_train_70 = []
test_train_80 = []
test_train_90 = []

# Fill each training set array with 10 sets of training data
for x in range(0, 10):
    test_train_1.append(train_test_split(features, class_label, test_size=0.99))
    test_train_2.append(train_test_split(features, class_label, test_size=0.98))
    test_train_3.append(train_test_split(features, class_label, test_size=0.97))
    test_train_4.append(train_test_split(features, class_label, test_size=0.96))
    test_train_5.append(train_test_split(features, class_label, test_size=0.95))
    test_train_6.append(train_test_split(features, class_label, test_size=0.94))
    test_train_7.append(train_test_split(features, class_label, test_size=0.93))
    test_train_8.append(train_test_split(features, class_label, test_size=0.92))
    test_train_9.append(train_test_split(features, class_label, test_size=0.91))
    test_train_10.append(train_test_split(features, class_label, test_size=0.90))
    test_train_20.append(train_test_split(features, class_label, test_size=0.80))
    test_train_30.append(train_test_split(features, class_label, test_size=0.70))
    test_train_40.append(train_test_split(features, class_label, test_size=0.60))
    test_train_50.append(train_test_split(features, class_label, test_size=0.50))
    test_train_60.append(train_test_split(features, class_label, test_size=0.40))
    test_train_70.append(train_test_split(features, class_label, test_size=0.30))
    test_train_80.append(train_test_split(features, class_label, test_size=0.20))
    test_train_90.append(train_test_split(features, class_label, test_size=0.10))

# Append all training set arrays to one array
test_train_arr = [test_train_1, test_train_2, test_train_3, test_train_4, test_train_5, test_train_6, test_train_7,
                  test_train_8, test_train_9, test_train_10, test_train_20, test_train_30, test_train_40, test_train_50,
                  test_train_60, test_train_70, test_train_80, test_train_90]

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
LinearRegression_model = LinearRegression(
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

SGDRegressor_model = SGDRegressor(
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

# Loop over each test_train_set
for test_train_set in test_train_arr:

    # Arrays to store classification accuracies for each test_train_set to calculate average accuracy
    dt_acc = []
    knn_acc = []
    mlp_acc = []

    # Arrays to store regression model error for each test_train_set to calculate average error
    lr_mae = []
    lr_mse = []
    lr_rmse = []
    sgd_mae = []
    sgd_mse = []
    sgd_rmse = []

    # Calculate size of current test_size as percent
    test_size = round((len(test_train_set[0][0]) / num_rows) * 100)

    # Loop over each set in the test_train_set
    for test_train in test_train_set:

        DecisionTreeClassifier_model.fit(test_train[0], test_train[2])
        KNN_model.fit(test_train[0], test_train[2])
        MLPClassifier_model.fit(test_train[0], test_train[2])
        LinearRegression_model.fit(test_train[0], test_train[2])
        SGDRegressor_model.fit(test_train[0], test_train[2])

        # Create Predictions
        DecisionTreeClassifier_prediction = DecisionTreeClassifier_model.predict(test_train[1])
        KNN_prediction = KNN_model.predict(test_train[1])
        MLPClassifier_prediction = MLPClassifier_model.predict(test_train[1])

        # Since actual values are 0 or 1, round to the nearest integer (0 or 1)
        LinearRegression_prediction = np.round(LinearRegression_model.predict(test_train[1]), 0)
        SGDRegressor_prediction = np.round(SGDRegressor_model.predict(test_train[1]), 0)

        # Print Classification Accuracy
        # print("\nDecision Tree Accuracy:         ", accuracy_score(DecisionTreeClassifier_prediction, test_train[3]))
        # print("Decision Tree Confusion Matrix: \n", confusion_matrix(DecisionTreeClassifier_prediction, test_train[3]))
        # print("K-Nearest Neighbour Accuracy:   ", accuracy_score(KNN_prediction, test_train[3]))
        # print("K-Nearest Neighbour Confusion Matrix: \n", confusion_matrix(KNN_prediction, test_train[3]))
        # print("Multilayer Perceptron Accuracy: ", accuracy_score(MLPClassifier_prediction, test_train[3]))
        # print("Multilayer Perceptron Confusion Matrix: \n", confusion_matrix(MLPClassifier_prediction, test_train[3]))

        # Print Regression Error
        # print('\nLinear Regression MAE:  ', metrics.mean_absolute_error(test_train[3], LinearRegression_prediction))
        # print('Linear Regression MSE:  ', metrics.mean_squared_error(test_train[3], LinearRegression_prediction))
        # print('Linear Regression RMSE: ', np.sqrt(metrics.mean_squared_error(test_train[3], LinearRegression_prediction)))
        # print('\nSGD Regression MAE:  ', metrics.mean_absolute_error(test_train[3], SGDRegressor_prediction))
        # print('SGD Regression MSE:  ', metrics.mean_squared_error(test_train[3], SGDRegressor_prediction))
        # print('SGD Regression RMSE: ', np.sqrt(metrics.mean_squared_error(test_train[3], SGDRegressor_prediction)))

        # Append classification accuracy to array to calculate average accuracy once the loop is done
        dt_acc.append(accuracy_score(DecisionTreeClassifier_prediction, test_train[3]))
        knn_acc.append(accuracy_score(KNN_prediction, test_train[3]))
        mlp_acc.append(accuracy_score(MLPClassifier_prediction, test_train[3]))

        # Append regression error to arrays to calculate averages once loop is done
        lr_mae.append(metrics.mean_absolute_error(test_train[3], LinearRegression_prediction))
        lr_mse.append(metrics.mean_squared_error(test_train[3], LinearRegression_prediction))
        lr_rmse.append(np.sqrt(metrics.mean_squared_error(test_train[3], LinearRegression_prediction)))
        sgd_mae.append(metrics.mean_absolute_error(test_train[3], SGDRegressor_prediction))
        sgd_mse.append(metrics.mean_squared_error(test_train[3], SGDRegressor_prediction))
        sgd_rmse.append(np.sqrt(metrics.mean_squared_error(test_train[3], SGDRegressor_prediction)))

        # Only create the plots if set to do so.
        if not do_plots:
            continue

        # Plot the first model.
        display_DecisionTreeClassifier = RocCurveDisplay.from_estimator(DecisionTreeClassifier_model, test_train[1], test_train[3])

        # For some reason, if we don't first make a dummy plot with just the first model, this will appear twice on the
        # later plot. Comment the below line and run to see what I mean. I can't seem to find a fix for this, and the
        # scikit docs example shows it this way as well.
        plt.show()

        # Plot the remaining models on top of the first model.
        ax = plt.gca()
        display_KNN = RocCurveDisplay.from_estimator(KNN_model, test_train[1], test_train[3], ax=ax, alpha=0.8)
        display_MLPClassifier = RocCurveDisplay.from_estimator(MLPClassifier_model, test_train[1], test_train[3], ax=ax, alpha=0.8)
        display_DecisionTreeClassifier.plot(ax=ax, alpha=0.8)

        # Show the comparison of all models.
        plt.show()

        # Give a moment to save the plot as something relevant as otherwise you will likely generate too much and lose
        # track of what plot goes with what.
        input('Hit enter to continue...')

    # Print average accuracy for classification models
    print("\n======================")
    print("Train ", test_size, "%")
    print("======================")

    print("\nD-T Avg Accuracy:   ", round((sum(dt_acc)/len(dt_acc))*100, 2), "%")
    print("KNN Avg Accuracy:   ", round((sum(knn_acc)/len(knn_acc))*100, 2), "%")
    print("MLP Avg Accuracy:   ", round((sum(mlp_acc)/len(mlp_acc))*100, 2), "%")

    # Print average error for regression models
    print("\nLin. Reg. Avg MAE:  ", round((sum(lr_mae)/len(lr_mae))*100, 4), "%")
    print("Lin. Reg. Avg MSE:  ", round((sum(lr_mse)/len(lr_mse))*100, 4), "%")
    print("Lin. Reg. Avg RMSE: ", round((sum(lr_rmse)/len(lr_rmse))*100, 4), "%")

    print("\nSDG Reg.  Avg MAE:  ", round((sum(sgd_mae)/len(sgd_mae))*100, 4), "%")
    print("SDG Reg.  Avg MSE:  ", round((sum(sgd_mse)/len(sgd_mse))*100, 4), "%")
    print("SDG Reg.  Avg RMSE: ", round((sum(sgd_rmse)/len(sgd_rmse))*100, 4), "%")