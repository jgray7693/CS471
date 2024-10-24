from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Import the iris dataset using sklearn load_iris function
iris_data = load_iris()
# Get the features and labels from the dataset
X = iris_data.data
y = iris_data.target

# Split the entire dataset into training and testing in an 80-20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Split the training dataset into training and validation, again in an 80-20 split
X_validation_train, X_validation_test, y_validation_train, y_validation_test = train_test_split(X_train, y_train, test_size=0.2)

from sklearn import tree
# Create a Decision Tree and fit it using the training dataset
default_clf = tree.DecisionTreeClassifier()
default_clf = default_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Define 2 hyperparameters - max_depth and min_samples_split
max_depth = [i for i in range(1, 6)]
min_samples_split = [i for i in range(2, 7)]

# Define a list to hold the results (max_depth, min_samples_split, accuracy)
results = []

# Define variables for optimal hyperparameters
best_accuracy = 0
optimal_depth = 0
optimal_split = 0

# Tune the 2 hyperparameters using the training and validation data
print('\nManually tuning hyperparameters max_depth and min_samples_split...\n')
for depth in max_depth:
    for samples in min_samples_split:
        # Initialize a Decision Tree using the defined hyperparameters
        validation_clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=samples)
        # Fit the Decision Tree with the validation adjusted training data
        validation_clf = validation_clf.fit(X_validation_train, y_validation_train)
        # Use the fitted Decision Tree to predict the classifications of the validation set
        prediction = validation_clf.predict(X_validation_test)
        # Determine the accuracy of our prediction
        accuracy = accuracy_score(y_validation_test, prediction)
        # Append the results of this runthrough to my results list
        results.append((depth, samples, accuracy))
        # If the results of this runthrough are the best so far, update all optimal hyperparameter variables
        if accuracy - best_accuracy > .001:
            best_accuracy = accuracy
            optimal_depth = depth
            optimal_split = samples

# print results of validation testing
print(f'Optimal max_depth: {optimal_depth}, Optimal min_sample_split: {optimal_split}, Best accuracy: {best_accuracy}\n')

# Plot the results where x is min_samples_split and y is accuracy. have 'depth' number of lines of differing color
plt.figure(figsize=(10, 6))
# Package groupings by min_samples_split, and assign data points to max_depth and accuracy
for sample in min_samples_split:
    split_results = [(depth, accuracy) for depth, s, accuracy in results if s == sample]
    split_results.sort(key=lambda x: x[0])  # Sort by min_samples_split
    depth, accuracies = zip(*split_results)
    plt.plot(depth, accuracies, marker='o', label=f'min_samples_split={sample}')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy for Different Hyperparameters')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import classification_report
# Fit a tuned Decision Tree using optimal hyperparameters with the entire training data, and test it on the unseen testing data
clf = tree.DecisionTreeClassifier(max_depth=optimal_depth, min_samples_split=optimal_split)
clf = clf.fit(X_train, y_train)
# Predict the testing data using the tuned Decision Tree and print the classification report
print('Iris classification using the tuned sklearn Decision Tree:\n')
final_prediction = clf.predict(X_test)
print(classification_report(y_test, final_prediction, target_names=iris_data.target_names))

# Plot the final Decision Tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, filled=True, feature_names=iris_data.feature_names, class_names=iris_data.target_names, fontsize=8)
plt.title('Decision Tree Trained on Iris Dataset')
plt.show()