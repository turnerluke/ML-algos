# Create Your Own k-Means Clustering Algorithm in Python

## Introduction

k-means clustering is an **unsupervised** machine learning algorithm that seeks to segment a dataset into groups based on the similarity of datapoints. An unsupervised model has **independent variables** and no **dependent variables**.

Suppose you have a dataset of 2-dimensional scalar attributes:

![data](https://github.com/turnerluke/ML-algos/blob/main/k_means/dataset.png)

If the points in this dataset belong to distinct groups with attributes significantly varying between groups but not within, the points should form clusters when plotted.

![Fig1](https://github.com/turnerluke/ML-algos/blob/main/k_means/unclassified%20example.png)  
**Figure 1:** *A dataset of points with groups of distinct attributes.*

This dataset clearly displays 3 distinct classes of data. If we seek to assign a new datapoint to one of these three groups, it can be done by finding the midpoint of each group (centroid) and selecting the nearest centroid as the group of the unassigned datapoint.

![Fig2](https://github.com/turnerluke/ML-algos/blob/main/k_means/classified%20example.png)  
**Figure 2:** *The datapoints are segmented into groups denoted with differing colors.*

## Algorithm

For a given dataset, k is specified to be the number of distinct groups the points belong to. These k centroids are first randomly initialized, then iterations are performed to optimize the locations of these k centroids as follows:

1. The distance from each point to each centroid is calculated.
2. Points are assigned to their nearest centroid.
3. Centroids are shifted to be the average value of the points belonging to it. If the centroids did not move, the algorithm is finished, else repeat.


## Data
To evaluate our algorithm, we'll first generate a dataset of groups in 2-dimensional space. The sklearn.datasets function make_blobs creates groupings of 2-dimensional normal distributions, and assigns a label corresponding to the group said point belongs to.

```
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

centers = 5
X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
X_train = StandardScaler().fit_transform(X_train)

sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                palette="deep",
                legend=None
                )

plt.xlabel("x")
plt.ylabel("y")
plt.show()
```
![Fig3](https://github.com/turnerluke/ML-algos/blob/main/k_means/Fig3.png)  
**Figure 3:** *The dataset we will use to evaluate our k means clustering model.*

This dataset provides a unique demonstration of the k-means algorithm. Observe the orange point uncharactaristically far from its center, and directly in the cluster of purple datapoints.
This point cannot be accurately classified as belonging to the right group, thus even if our algorithm works well it should incorreclty characterize it as a member of the purple group.

## Model Creation

### Helper Functions
We'll need to calculate the distances between a point and a dataset of points multiple times in this algorithm.
To do so lets define a function that calculates euclidean distances.
```
def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
```

### Implementation

First, the k-means clustering algorithm is initialized with a value for k and a maximum number of iterations for finding the optimal centroid locations. If a maximum number of iterations is not considered when optimizing centroid locations, there is a risk of running an infinite loop.

```
class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
```

Now, the bulk of the algorithm is performed when fitting the model to a training dataset.

First we'll initialize the centroids randomly in the domain of the test dataset, with a uniform distribution.

```
# Randomly select centroid start points, uniformly distributed across the domain of the dataset
min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
```

Next, we perform the iterative process of optimizing the centroid locations.

The optimization process is to readjust the centroid locations to be the means of the points belonging to it. This process is to repeat until the centroids stop moving, or the maximum number of iterations is passed. We'll use a while loop to account for the fact that this process does not have a fixed number of iterations. Additionally, you could also use a for loop that repeats max_iter times and breaks when the centroids stop changing.

Before begining the while loop, we'll initialize the variables used in the exit conditions.

```
iteration = 0
prev_centroids = None
```

Now, we begin the loop. We'll iterate through the datapoints in the training set, assigning them to an initialized empty list of lists. The sorted_points list contains one empty list for each centroid, where data points are appended once they've been asigned.

```
while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
    # Sort each datapoint, assigning to nearest centroid
    sorted_points = [[] for _ in range(self.n_clusters)]
    for x in X_train:
        dists = euclidean(x, self.centroids)
        centroid_idx = np.argmin(dists)
        sorted_points[centroid_idx].append(x)
```

Now that we've assigned the whole training dataset to their closest centroids, we can update the location of the centroids and finish the iteration.

```
    # Push current centroids to previous, reassign centroids as mean of the points belonging to them
    prev_centroids = self.centroids
    self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
    for i, centroid in enumerate(self.centroids):
        if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
            self.centroids[i] = prev_centroids[i]
    iteration += 1
```

After the completion of the iteration, the while conditions are checked again, and the algorithm will continue until the centroids are optimized or the max iterations are passed. The full fit method is included below.


```
class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
```

Lastly, lets make a method to evaluate a set of points to the centroids we've optimized to our training set. This method returns the centroid and the index of said centroid for each point.
   
 ```

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idx
```

### First Model Evalution

Now we can finally deploy our model. Lets train and test it on our original dataset and see the results. We'll keep our original method of plotting our data, by separating the true labels by color, but now we'll additionally separate the predicted labels by markerstyle, to see how the model performs.

```
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '+',
         markersize=10,
         )

plt.show()
```

![Fig4](https://github.com/turnerluke/ML-algos/blob/main/k_means/Fig4.png)  
**Figure 4:** *A failed example where one centroid has no points, and one contains two clusters.*

![Fig5](https://github.com/turnerluke/ML-algos/blob/main/k_means/Fig5.png)  
**Figure 5:** *A failed example where one centroid has no points, two contains two clusters, and two split one cluster.*

![Fig6](https://github.com/turnerluke/ML-algos/blob/main/k_means/Fig6.png)  
**Figure 6:** *A failed example where two centroids contain one and a half clusters, and two centroids splkit a cluster.*

### Re-evaluating Centroid Initialization

Looks like our model isn't performing very well. We can infer two primary problems from these three failed examples.
1. If a centroid is initialized far from any groups, it is unlikely to move. (Example: the bottom right centroid in **Figure 4**.
2. If centroids are initialized too close, they're unlikely to diverge from one another. (Example: the two centroids in the green group in **Figure 6**.

We'll begin to remedy these problems with a new process of initializing the centroid locations. This new method is referred to as the k-means++ algorithm.
1. Initialize the first centroid as a random selection of one of the data points.
2. Calculate the sum of the distances between each data point and all the centroids.
3. Select the next centroid randomly, with a probability proportional to the total distance to the centroids.
4. Return to step 2. Repeat until all centroids have been initialized.

This code is included below.

```
# Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
# then the rest are initialized w/ probabilities proportional to their distances to the first
# Pick a random point from train data for first centroid
self.centroids = [random.choice(X_train)]

for _ in range(self.n_clusters-1):
    # Calculate distances from points to the centroids
    dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
    # Normalize the distances
    dists /= np.sum(dists)
    # Choose remaining points based on their distances
    new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
    self.centroids += [X_train[new_centroid_idx]]
```

If we run this new model a few times we'll see it performs much better, but still not always perfect.

![Fig7](https://github.com/turnerluke/ML-algos/blob/main/k_means/Fig7.png)  
**Figure 7:** *An ideal convergence, after implementing the k-means++ initialization method.*

## Conclusion

And with that, we're finished. We learned a simple, yet elegant implementation of an unsupervised machine learning model. The complete project code is included below.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):

        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs


# Create a dataset of 2D distributions
centers = 5
X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
X_train = StandardScaler().fit_transform(X_train)

# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )

plt.show()
```

Thanks for reading!  
[Connect with me on LinkedIn](https://www.linkedin.com/in/turnermluke/)  
[See this project in GitHub](https://github.com/turnerluke/ML-algos/blob/main/k_means/k_means.py)
