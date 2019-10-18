# import the files in as CSVs. Going to be using 3 different datasets. Hip, wrist, and both
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV


hip = pd.read_csv('Hip_New.csv', engine ='c', low_memory=False)
wrist = pd.read_csv('Wrist_New.csv', engine='c', low_memory=False)
hip_and_wrist = pd.read_csv('Hip_and_Wrist_RT.csv', engine='c', low_memory=False)



# HIP DATA

# break the file up into the features and labels
# Labels are the values we want to predict
hip_labels = np.array(hip['RTExercise'])
# Remove all columns that will not be used as features. Only features are mean, sd, angle
# axis 1 refers to the columns
hip_features = hip.drop(['RTExercise', 'Joint'], axis=1)
# Saving feature names for later use
hip_feature_list = list(hip_features.columns)
print("Hip features being used are:", hip_feature_list)
# Convert to numpy array
hip_features = np.array(hip_features)


# Using Skicit-learn to split data into training and testing sets

# Split the data into training and testing sets
hip_train_features, hip_test_features, hip_train_labels, hip_test_labels = train_test_split(hip_features, hip_labels, test_size=0.25, random_state=42)

# print('Training Features Shape:', hip_train_features.shape)
# print('Training Labels Shape:', hip_train_labels.shape)
# print('Testing Features Shape:', hip_test_features.shape)
# print('Testing Labels Shape:', hip_test_labels.shape)


# Import the model we are using

# Instantiate model with 1000 decision trees
hip_rf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
print("done instantiating model")
# Train the model on training data
parameters = {
    'n_estimators': [1000, 1100, 1200, 1300]
    }
# clf = GridSearchCV(hip_rf, parameters, cv=5, verbose=1)
# clf.fit(hip_train_features, hip_train_labels)
#
# print(clf.best_params_)
# print(clf.best_score_)

hip_rf.fit(hip_train_features, hip_train_labels)
print("done fitting random forest")


# Use the forest's predict method on the test data
# hip_predictions = clf.predict(hip_test_features)
hip_predictions = hip_rf.predict(hip_test_features)
print("Accuracy score using hip data is: ", accuracy_score(hip_test_labels, hip_predictions))




# WRIST DATA

# break the file up into the features and labels
# Labels are the values we want to predict
wrist_labels = np.array(wrist['RTExercise'])
# Remove all columns that will not be used as features. Only features are mean, sd, angle
# axis 1 refers to the columns
wrist_features = wrist.drop(['RTExercise', 'Joint'], axis=1)
# Saving feature names for later use
wrist_feature_list = list(wrist_features.columns)
print("Wrist features being used are:", wrist_feature_list)
# Convert to numpy array
wrist_features = np.array(wrist_features)


# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
wrist_train_features, wrist_test_features, wrist_train_labels, wrist_test_labels = train_test_split(wrist_features, wrist_labels, test_size=0.25, random_state=42)

# Import the model we are using
# Instantiate model with 1000 decision trees
wrist_rf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
print("done instantiating model")
# Train the model on training data
wrist_rf.fit(wrist_train_features, wrist_train_labels)
print("done fitting random forest")

# Use the forest's predict method on the test data
wrist_predictions = wrist_rf.predict(wrist_test_features)
print("Accuracy score using wrist data is: ", accuracy_score(wrist_test_labels, wrist_predictions))




# HIP AND WRIST DATA

# break the file up into the features and labels
# Labels are the values we want to predict
hip_and_wrist_labels = np.array(hip_and_wrist['RTExercise'])
# Remove all columns that will not be used as features. Only features are mean, sd, angle
# axis 1 refers to the columns
hip_and_wrist_features = hip_and_wrist.drop(['RTExercise', 'Joint'], axis=1)
# Saving feature names for later use
hip_and_wrist_feature_list = list(hip_and_wrist_features.columns)
print("Hip and Wrist features being used are:", hip_and_wrist_feature_list)
# Convert to numpy array
hip_and_wrist_features = np.array(hip_and_wrist_features)


# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
hip_and_wrist_train_features, hip_and_wrist_test_features, hip_and_wrist_train_labels, hip_and_wrist_test_labels = train_test_split(hip_and_wrist_features, hip_and_wrist_labels, test_size=0.25, random_state=42)

# Import the model we are using
# Instantiate model with 1000 decision trees
# hip_and_wrist_rf = RandomForestClassifier(n_estimators = 1500, n_jobs=-1)
hip_and_wrist_rf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
print("done instantiating model")
# parameters = {
#     'n_neighbors': [4, 5, 6],
#     }
#
# clf = GridSearchCV(hip_and_wrist_rf, parameters, cv=5, verbose=1,n_jobs=-1)
# clf.fit(hip_and_wrist_train_features, hip_and_wrist_train_labels)
#
# print(clf.best_params_)
# print(clf.best_score_)

# # Train the model on training data
hip_and_wrist_rf.fit(hip_and_wrist_train_features, hip_and_wrist_train_labels)
print("done fitting random forest")

# Use the forest's predict method on the test data
# hip_and_wrist_predictions = clf.predict(hip_and_wrist_test_features)
hip_and_wrist_predictions = hip_and_wrist_rf.predict(hip_and_wrist_test_features)
print("Accuracy score using hip and wrist data is: ", accuracy_score(hip_and_wrist_test_labels, hip_and_wrist_predictions))




