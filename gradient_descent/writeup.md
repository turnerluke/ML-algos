# Create a Gradient Descent Algorithm with Regularization from Scratch in Python


## Introduction

Gradient descent is a fundamental algorithm used for machine learning and optimization problems. Thus, fully understanding its functions and limitations is critical for anyone studying machine learning or data science. This tutorial will implement a from-scratch gradient descent algorithm, test it on a simple model optimiztion problem, and lastly be adjusted to demonstrate parameter regularization.

## Background

Gradient descent seeks to find a local minimum of the **cost function** by adjusting model parameters. The **cost function** (or loss function) maps variables onto a real number representing a "cost" or value to be minimized.

For our model optimization, we'll perform **least squares optimization**, where we seek to minimize the sum of the differences between our predicted values, and the data values. **Equation 1** presents the cost function 


![Least Squares Cost Function](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ1.png)
**Equation 1:** *The least squares optimization cost function.*

Here, yhat is the model prediction from the independent variable. For this analysis we'll use a general polynomial model, presented in **Equation 2**.

![General Polynomial Model](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ2.png)
**Equation 2:** *The general polynomial model used in this analysis.*

For simplicity, we'll keep these equations in matrix form. Doing so presents our new model in **Equation 3** with the X matrix structure presented in **Equation 4**.

![Matrix Form](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ3.png)
**Equation 3:** *The matrix form of our model.*

![Polynomial X](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ4.png)
**Equation 4:** *The polynomial matrix, X.*

Now with this background of our cost function and the model we'll be deploying, we can now dive into gradient descent.

The chain rule (recall multivariable calculus) provides us with a method of approximating the change in cost for a given change in parameters. This relationship is presented in **Equation 5**.

![Change in Cost](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ5.png)
**Equation 5:** *The chain rule applied to determine changes in cost for changes in parameters.*

Knowing this, we can define a change in parameters to be proportional to the cost gradient, as presented in **Equation 6**. The learning rate (eta) is chosen to be a small, positive number.

![Parameter Update](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ6.png)
**Equation 6:** *Parameter update rules.*

When this rule to update parameters is plugged into **Equation 5**, we recieve our proof that the chosen parameter updating rule will always descend the cost.

![Cost Descent Proof](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ7.png)
**Equation 7:** *Proof the parameter updating rule will decrease the cost.*

If we recall linear algebra, we can remember that the square of the cost gradient vector will always be positive. Thus, provided the learning rate is small enough, this updating method will *descend the gradient* of the cost function.

Now, to finally implement this algorithm we need a method of numerically calculating the gradient. For this example we could sit with a pen and paper performing derivitives, however we'd like our algorithm to work for any model and cost function. **Equation 8** presents our method of doing so, where we will adjust each parameter by a small value and observe the change in cost.

![Numerical Calculation of Gradient](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ8.png)
**Equation 8:** *Numerical method of calculating the cost gradient.*

## Data
We'll generate our own dataset for this project. We'll simply generate a linearly spaced vector of independent values, and calculate the dependent variables from these, with some noise introduced. I've set my random seed to 42, to allow you to see if you get the same results.

```
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def polynomial_model(beta, x):
    '''
    A polynomial model.
    beta: numpy array of parameters of size (n,)
    x: numpy array of size (m,)

    return yhat: prediction of the model of size (m,)
    '''
    # Turn x (n,) to X (n, m) where m is the order of the polynomial
    # The second axis is the value of x**m
    X = x[:, np.newaxis] ** np.arange(0, len(beta))

    # Perform model prediction
    yhat = np.sum(beta * X, axis=1)

    return yhat

# Construct a dataset
x = np.linspace(-2, 2, 5)
beta_actual = [3, 2, 1/3]
y = polynomial_model(beta_actual, x) + np.random.normal(size=x.size, scale=0.5)
```

I've chosen the actual model parameters to be [3, 2, 1/3], and the noise to be a normal distribution with standard deviation of 0.5. Lets view the data below.

```
# Plot results
fig, ax = plt.subplots()
ax.plot(x, y, '.')
xplt = np.linspace(min(x), max(x), 100)
yplt = polynomial_model(beta_actual, xplt)
plt.plot(xplt, yplt, '-')

ax.legend(['Data', 'Actual Relationship'])

plt.show()
```


![Data & Actual Model](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Fig1.png)
**Figure 1:** *Our data and actual model.*

## Model Creation

### Functions

Our cost function is defined below. Notice we'll only make beta to be a positional, the rest we'll pass with keyword arguments. This is to improve readability of our final gradient descent algorithm, which we'll see later.
```
def cost(beta, **kwargs):
    """
    Calculates the quadratic cost, with an optional regularization
    :param beta: Model Parameters
    :param kwargs:
    :return:
    """
    x = kwargs['x']
    y = kwargs['y']
    model = kwargs['model']

    # Calculate predicted y given parameters
    yhat = model(beta, x)

    # Calculate the cost
    C = sum((y-yhat)**2) / len(y)
    return C
```

### Algorithm

Our gradient descent class requires our model, cost function, an initial parameter guess, and our data. We could also tweak parameters like the learning rate or step in parameters to calculate the gradient, but for this analysis I set them to be sufficiently small numbers and did not optimize their values.
```
class GradDescent:

    def __init__(self, model, C, beta0, x, y, dbeta=1E-8, eta=0.0001, ftol=1E-8):
        self.model = model
        self.C = C
        self.beta = beta0
        self.x = x
        self.y = y
        self.dbeta = dbeta
        self.eta = eta
        self.ftol = ftol
```

Now we can finally implement the gradient descent algorithm. We'll begin by creating a dictionary of inputs to our cost function that do not change by iteration.

```
    def descend(self):
        # This dict of cost parameters does not change between calls
        cost_inputs = {'x': self.x,
                       'y': self.y,
                       'model': self.model
                       }
```
Next we'll initialize a list of costs, and begin iterating.
```
        # Initialize a list of costs, with the indices being the iteration
        costs = [self.C(self.beta, **cost_inputs)]

        run_condition = True
        
```
For each iteration we must:
1. Calculate the gradient
2. Update the parameters
3. Calculate the new cost
4. Evaluate our running condition

This process is presented below
```
        while run_condition:
            # Get the gradient of the cost
            delC = []

            for n, beta_n in enumerate(self.beta):
                # Create a temporary parameters vector, to change the nth parameter
                temp_beta = self.beta
                temp_beta[n] = beta_n + self.dbeta  # Adjusts the nth parameter by dbeta
                C_n = self.C(temp_beta, **cost_inputs)
                dC = C_n - costs[-1]
                delC.append(dC / self.dbeta)

            # Update the parameters
            self.beta = self.beta - self.eta * np.array(delC)

            # Re calc C
            costs.append(self.C(self.beta, **cost_inputs))

            # Evaluate running condition
            run_condition = abs(costs[-1] - costs[-2]) > self.ftol
```

Now we're ready to implement our model. Lets initialize starting parameters, create a gradient descent object, optimize our model and plot the results.

```
# Initialize parameters, use a polynomial of order 5
beta0 = np.random.normal(size=(5,), scale=0.1)

# Initialize a GradDescent object, perform descent and get parameters
gd = GradDescent(polynomial_model, cost, beta0, x, y)
gd.descend()

beta = gd.beta

# Make model prediction with parameters
yhat = polynomial_model(beta, x)

# Plot results
fig, ax = plt.subplots()
ax.plot(x, y, '.')
ax.plot(x, yhat, 'x')
xplt = np.linspace(min(x), max(x), 100)
yplt = polynomial_model(beta_actual, xplt)
plt.plot(xplt, yplt, '-')
yplt = polynomial_model(beta, xplt)
plt.plot(xplt, yplt, '--')

ax.legend(['Data', 'Predicted Values', 'Actual Relationship', 'Predicted Model'])

plt.show()
```

![Model Fit No Regularization](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Fig2.png)
**Figure 2:** *4th order polynomial fit of our data.*

Notice our model is intentionally overfitted. We have a 4th order polynomial fit to 5 data points, recall that an nth order polynomial can always perfectly predict n+1 data points, without any consideration to the underlying model.

Lets slightly modify our cost function to penalize the size of parameters. This process is referred to as **regularization** defined as the process of adding information in order to solve an ill-posed problem to prevent overfitting. We'll perform two types of regularization, **L1** or **Lasso Regression** (Least absolute shrinkage and selection operator) and **L2** or **Ridge Regression**. The modified cost functions for these techniques are presented below in **Equations 9 & 10**.

![L1/Lasso](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ9.png)
**Equation 9:** *Cost function for L1 Regularization.*

![L2/Ridge](https://github.com/turnerluke/ML-algos/blob/main/gradient_descent/Equ10.png)
**Equation 10:** *Cost function for L2 Regularization.*


We can easily modify our code to handle these regularization techniques. The only changes will occur in the cost function and the GradDescent object, shown below.

```
class GradDescent:

    def __init__(self, model, C, beta0, x, y, reg=None, lmda=0, dbeta=1E-8, eta=0.0001, ftol=1E-8):
        self.model = model
        self.C = C
        self.beta = beta0
        self.x = x
        self.y = y
        self.reg = reg
        self.lmda = lmda
        self.dbeta = dbeta
        self.eta = eta
        self.ftol = ftol


    def descend(self):
        # This dict of cost parameters does not change between calls
        cost_inputs = {'x': self.x,
                       'y': self.y,
                       'reg': self.reg,
                       'lmda': self.lmda,
                       'model': self.model
                       }
        # Initialize a list of costs, with the indices being the iteration
        costs = [self.C(self.beta, **cost_inputs)]

        run_condition = True
        while run_condition:

            # Get the gradient of the cost
            delC = []

            for n, beta_n in enumerate(self.beta):
                # Create a temporary parameters vector, to change the nth parameter
                temp_beta = self.beta
                temp_beta[n] = beta_n + self.dbeta  # Adjusts the nth parameter by dbeta
                C_n = self.C(temp_beta, **cost_inputs)
                dC = C_n - costs[-1]
                delC.append(dC / self.dbeta)

            # Update the parameters
            self.beta = self.beta - self.eta * np.array(delC)

            # Re calc C
            costs.append(self.C(self.beta, **cost_inputs))

            # Evaluate running condition
            run_condition = abs(costs[-1] - costs[-2]) > self.ftol

def cost(beta, **kwargs):
    """
    Calculates the quadratic cost, with an optional regularization
    :param beta: Model Parameters
    :param kwargs:
    :return:
    """
    x = kwargs['x']
    y = kwargs['y']
    reg = kwargs['reg']
    lmda = kwargs['lmda']
    model = kwargs['model']

    # Calculate predicted y given parameters
    yhat = model(beta, x)

    # Calculate the cost
    C = sum((y-yhat)**2) / len(y)
    if reg is not None:
        if reg == 'L1':  # For Lasso Regression (L1), add the magnitudes
            C += lmda * sum(abs(beta))
        elif reg == 'L2':  # For Ridge Regression (L2), add the squared magnitude
            C += lmda * sum(beta**2)
    return C
```

