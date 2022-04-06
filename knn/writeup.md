# Create your own k-Nearest Neighbors algorithm in Python

## Introduction
The k-nearest neighbors (knn) algorithm is a supervised learning algorithm with an elegant execution and a surprisingly easy implementation.
Because of this, knn presents a great learning opportunity for machine learning beginners to create a powerful classification or regression algorithm, with a few lines of Python code.

## Algorithm

Suppose you have a dataset of scalar attributes $(X_1^1, X_2^1), (X_1^2, X_2^2), ..., (X_1^n, X_2^n)$ and classes corresponding to said attributes $Y^1, Y^2, ..., Y^n$ where $Y \in {1, 2, ..., m}$. Here, n is the total number of data points and m is the total number of classes.

With this two dimensional example we can easily visualize these points on two-dimensional space. Assuming that classes will tend to cluster with points of the same class in this space, we can classify a new point by the most frequently occuring class near it. Thus, at a given point k is specified as the number of neighbors to consider near said point, and from those neighbors the most frequently occuring class is predicted to be the class of the point at hand.

# Put some descriptive plots here


## Data
We'll evaluate our algorithm with the UCI Machine Learning Repository iris dataset. However, any classification dataset consisting of scalar inputs will do. We'll unpack the dataset, and standardize the attributes to have no mean and unit variance. This is done because we don't want to pass any judgements on which features are most important for predicting the class (for this analysis!).

```
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

We'll need to figure out the most commonly occuring element in a given group at numerous times in this algorithm. The following function presents an intuitive and quite Pythonic method of finding the most common element in a list.

```
def most_common(lst):
    return max(set(lst), key=lst.count)
```

Next, we'll need to calculate the distance between a point and every point in a dataset. The most common and intuitive method of doing this is by euclidean distance, however other distance methods can be used.


```
def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
```

Now, lets begin to construct a knn class. For a given knn classifier, we'll specify k and a distance metric. To keep the implementation of this algorithm similar to that of the widely-used scikit-learn suite, we'll initialize the self.X_train and self.y_train in a fit method, however this could be done on initialization.

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
