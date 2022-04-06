# Introduction
The k-nearest neighbors (knn) algorithm is a supervised learning algorithm with an elegant execution and a surprisingly easy implementation.
Because of this, knn presents a great learning opportunity for machine learning beginners to create a powerful classification or regression algorithm, with a few lines of Python code.

# Data
We'll evaluate our algorithm with the UCI Machine Learning Repository iris dataset. However, any classification dataset consisting of scalar inputs will do.

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

