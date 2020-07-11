# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump

# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# model = svm.SVC(gamma=0.001)

# # Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=222)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=222)

# # We learn the digits on the first half of the digits
# model.fit(X_train, y_train)

# # Now predict the value of the digit on the second half:
# print(model.score(X_val, y_val))
print(X_val[0])
print(y_val[0])

# dump(model, 'numbers_model.joblib') 
