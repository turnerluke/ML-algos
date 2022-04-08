# ML-algos

A collection of homemade machine learning algorithmns. Algorithms are created with the intent of better understanding the processes of these ML algos.

All implementations are made to mirror the functionality of the scikit-learn algos.

## k-Nearest Neighbors Classifier
A from-scratch implementation fo the k-nearest neighbors classifier algorithm. Implemented using euclidean distance calculations, but easily generalizes to any other distance metric.

Tested on the UCI ML Repository iris dataset, performing above 95% accuracy at properly tuned k values. To view the relationship between k and accuracy of this algorithm, please see the attached graph, knn.png.

## k-Nearest Neighbors Regressor
An adaptation of the knn classification algorithm to handle regressions.

Tested on the StatLib repository Claifornia housing dataset.

## k-Means Clustering
A from-scratch implementation of the k-means clustering algorithm. Utilizes the k-means++ algorithm for choosing initial centroid values.

Tested on a generated dataset of 2-dimensional distributions normalized with standard scaling. The random seed is set in the script as the results display a data point that will be incorrectly characterized. The data and results are visualized in the scatterplot k-means.png. True labels are separted by color and predicted labels are denoted by shape, thus displaying points that are incorrectly characterized.

## Neural Networks

Special thanks to [Michael Nielsen](https://michaelnielsen.org/) for writing the exceptional ebook: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). This book provided a great learning experience of neural networks, and was a preliminary inspiration for my transition into a computational career.

### Multi-Layer Perceptron Classifier (MLPClassifier)
Implementation of a MLP Classifier neural network, trains with mini-batch stochastic gradient descent.

Tested on the MNIST digit dataset. Results in approximately 90% accuracy (may vary) with minimal parameter tuning.

### Multi-Layer Perceptron Regressor (MLPRegressor)
Implementation of a MLP Regressor neural network, trains with mini-batch stochastic gradient descent.
