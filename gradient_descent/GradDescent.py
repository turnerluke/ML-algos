
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)


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



def polynomial_model(beta, x):
    '''
    A polynomial model.
    beta: numpy array of parameters of size (m,)
    x: numpy array of size (n,)

    return yhat: prediction of the model of size (n,)
    '''
    # Turn x (n,) to X (n, m) where m is the order of the polynomial
    # The second axis is the value of x**m
    X = x[:, np.newaxis] ** np.arange(0, len(beta))

    # Perform model prediction
    yhat = np.sum(beta * X, axis=1)

    return yhat


def cost(beta, **kwargs):
    """
    Calculates the quadratic cost, with an optional regularization
    beta: Model Parameters

    return: Cost
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


# Construct a dataset
x = np.arange(-2, 3)
beta_actual = [1, 2, 1]
y = polynomial_model(beta_actual, x) + np.random.normal(size=x.size, scale=1)


# Initialize parameters, use a polynomial of order 4
beta0 = np.random.normal(size=(5,), scale=1)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (reg, lmda) in enumerate(zip([None, 'L1', 'L2'], [0, 1, 1])):
    # Initialize a GradDescent object, perform descent and get parameters
    gd = GradDescent(polynomial_model, cost, beta0, x, y, reg=reg, lmda=lmda)
    gd.descend()

    beta = gd.beta

    # Make model prediction with parameters
    yhat = polynomial_model(beta, x)

    axs[i].plot(x, y, '.')
    axs[i].plot(x, yhat, 'x')
    xplt = np.linspace(min(x), max(x), 100)
    yplt = polynomial_model(beta_actual, xplt)
    axs[i].plot(xplt, yplt, '--')
    yplt = polynomial_model(beta, xplt)
    axs[i].plot(xplt, yplt, '--')

    # Set title
    if reg is not None:
        axs[i].set_title(reg)
    else:
        axs[i].set_title("No Regularization")

    # Clean up the plots - remove x,y ticks and labels
    axs[i].axes.xaxis.set_ticklabels([])
    axs[i].axes.yaxis.set_ticklabels([])
    axs[i].axes.xaxis.set_visible(False)
    axs[i].axes.yaxis.set_visible(False)


fig.legend(['Data', 'Predicted Values', 'Actual Relationship', 'Predicted Model'])
plt.show()

