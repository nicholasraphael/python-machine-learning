# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#set up data frame using pandas.read_csv
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#look at the data
print(dataset.shape)
print(dataset.head(20))

#statistical summary
print(dataset.describe())

#class distribution/number in each class of flower
print(dataset.groupby('class').size())

#visualizations:

#univariate/plots of each individual variable
# histograms
#dataset.hist()
#plt.show()

#multivariate/interactions between the variables
#scatter plot
#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4] #values/input
Y = array[:,4] #labels/output
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

#possible models :
#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes .
#Support Vector Machines (SVM).

#K-nearest neighbors is best model for this dataset

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
