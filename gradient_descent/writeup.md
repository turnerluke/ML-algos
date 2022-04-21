# Create a Gradient Descent Algorithm from Scratch in Python


## Introduction

Gradient descent is a fundamental algorithm used for machine learning and optimization problems. Thus, fully understanding its functions and limitations is critical for anyone studying machine learning or data science. This tutorial will implement a from-scratch gradient descent algorithm, test it on a simple model optimiztion problem, and lastly be adjusted to demonstrate parameter regularization.

## Algorithm

Gradient descent seeks to find a local minimum of the **cost function** by adjusting model parameters. The **cost function** (or loss function) maps variables onto a real number representing a "cost" or value to be minimized.

For our model optimization, we'll perform **least squares optimization**, where we seek to minimize the sum of the differences between our predicted values, and the data values.


**Equation 1:** 
