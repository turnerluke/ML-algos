# Create your own k-Nearest Neighbors algorithm in Python

## Introduction
The k-nearest neighbors (knn) algorithm is a supervised learning algorithm with an elegant execution and a surprisingly easy implementation.
Because of this, knn presents a great learning opportunity for machine learning beginners to create a powerful classification or regression algorithm, with a few lines of Python code.

## Algorithm

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
```
def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
```


```
class KNeighborsClassifier():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
```
