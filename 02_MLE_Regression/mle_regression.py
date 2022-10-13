# import libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize, optimize
import scipy.stats as statsimport
# import pymc3 as pm3
# import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


def precission(y, yhat):
    res = []
    for i in range(yhat):
        res.append((y[i] - yhat[i]) ** 2)

    return res


# define likelihood function
def MLERegression(params):
    intercept, beta, sd = params[0], params[1], params[2]  # inputs are guesses at our parameters
    yhat = intercept + beta * x  # predictions# next, we flip the Bayesian question'
    # print(yhat)
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(scipy.stats.norm.logpdf(y, loc=yhat, scale=sd))  # return negative LL

    return (negLL)


# generate data
N = 20
x = np.linspace(0, 19, N)
# np.random.seed(1231124)
noise = np.random.normal(loc=0.0, scale=1.0, size=N)

y = 6 * x + 3 + noise

df = pd.DataFrame({'x': x, 'y': y})

df['constant'] = 0

# plt.scatter(df.x, df.y)
# plt.show()

# split features and target
X = df[['constant', 'x']]  # fit model and summarize

# print(sm.OLS(y, X).fit().summary())  # 1-й метод - МНК (статистическая оценка) - минимизировали сумму расстояний

# # # let’s start with some random coefficient guesses and optimize
# # 2-й метод - метод максимимального правдоподобия (байесовская оценка) - максимизировали вероятность
# # того, что точки принадлежат данному распределению
# guess = np.array([0, 0, 0])
#
# results = minimize(MLERegression, guess, method='Nelder-Mead', options={'disp': True})
# print(results.x)
#
# y_b = []
# z_k = []
fprime = lambda x: optimize.approx_fprime(x, MLERegression, 0.01)

b_plus_k = []
number_of_iter = []
for b in range(10):
    for k in range(10):
        for n in range(10):
            results = minimize(MLERegression, np.array([b, k, n]), method='Nelder-Mead', options={'disp': False},
                               jac=fprime, hess=fprime)
            number_of_iter.append(results.nit)
            b_plus_k.append(sum(results.x))

x = np.linspace(0, 1000, 1000)
df = pd.DataFrame({'x': x, 'y': b_plus_k})
estimated_res = 10

y = 10
plt.plot(x, [y] * len(x), color='green')
plt.scatter(df.x, df.y, s=4)
plt.xlabel('x - a combination of initial guess params')
plt.ylabel('y - sum of outputted params')
plt.show()

df = pd.DataFrame({'x': x, 'y': number_of_iter})
plt.scatter(df.x, df.y, s=4)
plt.show()
