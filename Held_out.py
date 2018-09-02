#Held out validation for galaxy redshift prediction
#50:50 split for the training and testing data
import numpy as np
from sklearn.tree import DecisionTreeRegressor


# function to assign the features and targets
def get_features_targets(data):
    features = np.zeros(shape=(len(data), 4))  # declare features
    features[:, 0] = data['u'] - data['g']
    features[:, 1] = data['g'] - data['r']
    features[:, 2] = data['r'] - data['i']
    features[:, 3] = data['i'] - data['z']
    targets = data['redshift']  # assign targets
    return features, targets




# function to calculate the median difference between the actual and predicted redshifts
def median_diff(predicted, actual):
    med_diff = np.median(np.abs(predicted[:] - actual[:]))
    return med_diff


#function to perform held out validation using a 50:50 split
#and train a decision tree model to predict redshift values
#The prediction accuracy is returned with median_diff
def validate_model(model, features, targets):
    # split the data into training and testing features and predictions
    split = features.shape[0] // 2
    train_features = features[:split]
    test_features = features[split:]
    splitt = targets.shape[0] // 2
    train_targets = targets[:splitt]
    test_targets = targets[splitt:]

    # train the model
    model.fit(train_features, train_targets)
    # get the predicted_redshifts
    predictions = model.predict(test_features)

    # use median_diff function to calculate the accuracy
    return median_diff(test_targets, predictions)


if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    features, targets = get_features_targets(data)

    # initialize model
    dtr = DecisionTreeRegressor()

    # validate the model and print the med_diff
    diff = validate_model(dtr, features, targets)
    print('Median difference: {:f}'.format(diff))
