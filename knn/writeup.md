# Create your own k-Nearest Neighbors Algorithm in Python

## Introduction
A famous quote states: "You are the average of the five people you spend the most time with." Although we won't be modelling the qualities of your friendships (portfolio project anyone?), this tutorial will teach a simple and intuitive algorithmic approach to classifying data based on their neighbors. The k-nearest neighbors (knn) algorithm is a supervised learning algorithm with an elegant execution and a surprisingly easy implementation. Because of this, knn presents a great learning opportunity for machine learning beginners to create a powerful classification or regression algorithm, with a few lines of Python code.

## Algorithm

Knn is a **supervised** machine learning algorithm. A supervised model has both a **target variable** and **independent variables**. The **target variable** or dependent variable, denoted y, depends on the independent variables and is the value you seek to predict. The **independent variables**, denoted x (single valued) or X (multi valued), are known ahead of time and are used to predict y.

Knn can be used for both **clasification* and **regression**. **Classification** models predict a *categorical* target variable and **regression** models predict a *numeric* target.

Suppose you have a dataset of scalar attributes $(X_1^1, X_2^1), (X_1^2, X_2^2), ..., (X_1^n, X_2^n)$ and classes corresponding to said attributes $y^1, y^2, ..., y^n$ where $y \in {1, 2, ..., m}$. Here, n is the total number of data points and m is the total number of classes. Insted of y being a class, it could also be a scalar value, and knn can be used as a regression, but for this tutorial we will focus on classification.

With this two dimensional example we can easily visualize these points in two-dimensional space. Assuming that classes will tend to cluster with points of the same class in this space, we can classify a new point by the most frequently occuring class near it. Thus, at a given point k is specified as the number of neighbors to consider near said point, and from those neighbors the most frequently occuring class is predicted to be the class of the point at hand.

![k=1](https://github.com/turnerluke/ML-algos/blob/main/knn/knn1.png)
**Figure 1:** *The point is classified as group 1 when k=1.*


![k=3](https://github.com/turnerluke/ML-algos/blob/main/knn/knn3.png)
**Figure 2:** *The point is classified as group 0 when k=3.*



## Data
We'll evaluate our algorithm with the UCI Machine Learning Repository iris dataset. However, any classification dataset consisting of scalar inputs will do. We'll unpack the dataset, and standardize the attributes to have zero mean and unit variance. This is done because we don't want to pass any judgements on which features are most important for predicting the class (for this analysis!). Lastly, we'll split our dataset into training and testing sets, with the test set consisting of 20% of the original dataset.

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Unpack the iris dataset, from UCI Machine Learning Repository
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Preprocess data
X = StandardScaler().fit_transform(X)

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## Model Creation

### Helper Functions

We'll need to figure out the most commonly occuring element in a given group at numerous times in this algorithm. The following function presents an intuitive and  Pythonic method of finding the most common element in a list.

```
def most_common(lst):
    '''Returns the most common element in a list'''
    return max(set(lst), key=lst.count)
```

Next, we'll need to calculate the distance between a point and every point in a dataset. The most common and intuitive method of doing this is by euclidean distance, however other distance methods can be used.


```
def euclidean(point, data):
    '''Euclidean distance between a point  & data'''
    return np.sqrt(np.sum((point - data)**2, axis=1))
```
# Adjust this to new descriptions


### Implementation

Now, lets begin to construct a knn class. For a given knn classifier, we'll specify k and a distance metric. To keep the implementation of this algorithm similar to that of the widely-used scikit-learn suite, we'll initialize the self.X_train and self.y_train in a fit method, however this could be done on initialization.

```
class KNeighborsClassifier():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
```

Next, the bulk of the knn algorithm is performed by the prediction method. A dataset of attributes (X_test) is iterated through, point by point. For each datapoint the following steps are performed:
1. Distances to every point in the training dataset is calculated
2. The training dataset classes are sorted by distance to the data point
3. The first k classes are kept and stored in the neighbors list
Now we simply map the list of nearest neighbors to our most_common function, returning a list of predictions for each point passed in X_test.

```
class KNeighborsClassifier():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return list(map(most_common, neighbors))
```
Lastly, an evaluate method is defined to conveniently evaluate the performance of our model. A dataset of attributes and their classes are passed as X_test and y_test, and the predictions of the model from the attributes are compared to the actual classes.

```
class KNeighborsClassifier():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy
```

Believe it or not, we're finished- we can easily deploy this algorithm to model classificaiton problems. But, for completeness we should optimize k for the iris dataset. We can do this by iterating through a range of k and plotting the performance of the model.
```
accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)
    
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()
```

![knn performance](https://github.com/turnerluke/ML-algos/blob/main/knn/knn.png?raw=true)
**Figure 3:** *knn accuracy versus k*

Looks like our knn model performs best at low k.

## Conclusion

And with that we're done. We've implemented a simple and intuitive k-nearest neighbors algorithm with under 100 lines of python code (under 50 excluding the plotting and data unpacking). The entire project code is included below.
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


# Unpack the iris dataset, from UCI Machine Learning Repository
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess data
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

# Test knn model across varying ks
accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()
```
Returning to our quote ""You are the average of the five people you spend the most time with", knn classification should instead say "You are the most frequent of the k people you spend the most time with." For an independent challenge adapt this code to better fit the original code by creating a knn regression model, where a point is interpreted as the average scalar target value of its k nearest neighbors.

Thanks for reading!
[Connect with me on LinkedIn](https://www.linkedin.com/in/turnermluke/)
[See this project in GitHub](https://github.com/turnerluke/ML-algos/blob/main/knn/KNeighborsClassifier.py)
[See a knn regression implementation](https://github.com/turnerluke/ML-algos/blob/main/knn/KNeighborsRegressor.py)
