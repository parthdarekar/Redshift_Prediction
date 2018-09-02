#To find out the maximum depth of the decision tree to minimise the error for the testing set
#Also observe how the error(median difference) varies with depth for training and testing sets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor


#assigning the features and targets
def get_features_targets(data):
    features = np.zeros((data.shape[0], 4))

    features[:, 0] = data['u'] - data['g']
    features[:, 1] = data['g'] - data['r']
    features[:, 2] = data['r'] - data['i']
    features[:, 3] = data['i'] - data['z']
    targets = data['redshift']
    return features, targets


# calculating the median difference
def median_diff(predicted, actual):
    med_diff = np.median(np.abs(predicted[:] - actual[:]))
    return med_diff


# Calculating the median difference for decision trees with different maximum depths
def accuracy_by_treedepth(features, targets, depths):
    # split the data into testing and training sets
    split = features.shape[0] // 2
    training_features, test_features = features[:split], features[split:]
    training_targets, test_targets = targets[:split], targets[split:]

    # initialise arrays or lists to store the accuracies for the below loop
    accuracy_training = []
    accuracy_test = []
    # loop through depths
    for depth in depths:
        
        # initialize model with the maximum depth.
        dtr = DecisionTreeRegressor(max_depth=depth)

        # train the model using the training set
        dtr.fit(training_features, training_targets)

        # get the predictions for the training set and calculate their median_diff
        training_predictions = dtr.predict(training_features)
        accuracy_training.append(median_diff(training_predictions, training_targets))

        # get the predictions for the testing set and calculate their median_diff
        test_predictions = dtr.predict(test_features)
        accuracy_test.append(median_diff(test_predictions, test_targets))

    # return the accuracies for the training and testing sets
    return accuracy_training, accuracy_test


if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    features, targets = get_features_targets(data)

    # Generate several depths to test
    tree_depths = [i for i in range(1, 36, 2)]

    # Call the function
    train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
    print("Depth with lowest median difference : {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))

    # Plot the results
    train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
    test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Median of Differences")
    plt.legend()
    plt.show()


