# Cross Validation Classification Accuracy

# 1. Classification Accuracy
# Classification accuracy is the number of correct predictions made as a ratio of all predictions made.
#
# This is the most common evaluation metric for classification problems, it is also the most misused.
# It is really only suitable when there are an equal number of observations in each class (which is rarely the case)
# and that all predictions and prediction errors are equally important, which is often not the case.
#
# Below is an example of calculating classification accuracy.


import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)

dataframe.to_csv('pima.csv')
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# 2. Log Loss
# Logistic loss (or log loss) is a performance metric for evaluating
# the predictions of probabilities of membership to a given class.
# The scalar probability between 0 and 1 can be seen as a measure of confidence for a prediction by an algorithm.
# Predictions that are correct or incorrect are rewarded or punished proportionally
# to the confidence of the prediction.
#
# Below is an example of calculating log loss for Logistic regression predictions
# on the Pima Indians onset of diabetes dataset.
# Smaller log loss is better with 0 representing a perfect log loss.

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#
# 3. Area Under ROC Curve
# Area Under ROC Curve (or ROC AUC for short) is a performance metric for binary classification problems.
#
# The AUC represents a model’s ability to discriminate between positive and negative classes.
# An area of 1.0 represents a model that made all predictions perfectly.
# An area of 0.5 represents a model as good as random.
#
# A ROC Curve is a plot of the true positive rate and the false positive rate
# for a given set of probability predictions at different thresholds used to map the probabilities to class labels.
# The area under the curve is then the approximate integral under the ROC Curve.
#
# The example below provides a demonstration of calculating AUC.

# Cross Validation Classification ROC AUC
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#
# 4. Confusion Matrix
# The confusion matrix is a handy presentation of the accuracy of a model with two or more classes.
#
# The table presents predictions on the x-axis and accuracy outcomes on the y-axis.
# The cells of the table are the number of predictions made by a machine learning algorithm.
#
# For example, a machine learning algorithm can predict 0 or 1 and each prediction
# may actually have been a 0 or 1.
# Predictions for 0 that were actually 0 appear in the cell for prediction=0 and actual=0,
# whereas predictions for 0 that were actually 1 appear in the cell for prediction = 0 and actual=1. And so on.

# Below is an example of calculating a confusion matrix for a set of prediction by a model on a test set.

# Cross Validation Classification Confusion Matrix
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

# 5. Classification Report
# Scikit-learn does provide a convenience report when working on classification problems
# to give you a quick idea of the accuracy of a model using a number of measures.
#
# The classification_report() function displays the precision, recall, f1-score and support for each class.
#
# The example below demonstrates the report on the binary classification problem.

# Cross Validation Classification Report
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)

# Regression Metrics
# In this section will review 3 of the most common metrics for evaluating predictions on regression machine learning problems:
#
# Mean Absolute Error.
# Mean Squared Error.
# R^2.
# 1. Mean Absolute Error
# The Mean Absolute Error (or MAE) is the average of the absolute differences
# between predictions and actual values. It gives an idea of how wrong the predictions were.
#
# The measure gives an idea of the magnitude of the error,
# but no idea of the direction (e.g. over or under predicting).
#
# The example below demonstrates calculating mean absolute error on the Boston house price dataset.


# Cross Validation Regression MAE
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))

# 2. Mean Squared Error
# The Mean Squared Error (or MSE) is much like the mean absolute error in that it provides
# a gross idea of the magnitude of error.
#
# Taking the square root of the mean squared error converts
# the units back to the original units of the output variable and can be meaningful for description and presentation.
# This is called the Root Mean Squared Error (or RMSE).

# Cross Validation Regression MSE
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))

# 3. R^2 Metric
# The R^2 (or R Squared) metric provides an indication of the goodness of
# fit of a set of predictions to the actual values.
# In statistical literature, this measure is called the coefficient of determination.
#
# This is a value between 0 and 1 for no-fit and perfect fit respectively.
#
# The example below provides a demonstration of calculating the mean R^2 for a set of predictions.

# Cross Validation Regression R^2
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))
