import numpy as np
import pandas as pd
import random


def cost_function(theta):
    return sum(theta_i ** 2 for theta_i in theta)


def differnce_quotient(f, theta, i, h=0.0001):
    w = [theta_j + (h if j == i else 0) for j, theta_j in enumerate(theta)]
    return (f(w) - f(theta)) / h


# def gradient(x):
#     return sum(2 * x_i for x_i in x)
tmp = [differnce_quotient(cost_function, [1, 2, 3], i) for i in range(3)]
f = cost_function
theta = [1, 2, 3, 4]
np.array([differnce_quotient(f, theta, j) for j in range(4)])


def gradient_descent(f, learning_rate=0.01, tolerance=0.000001):
    theta = np.array([random.randint(-10, 10) for i in range(4)])
    # theta = [1, 2, 3, 4]
    while True:
        gradient = np.array([differnce_quotient(f, theta, j) for j in range(4)])
        next_theta = theta - gradient * learning_rate
        if sum(np.sqrt((next_theta - theta) ** 2)) < tolerance:
            return next_theta
        theta = next_theta


gradient_descent(cost_function)
