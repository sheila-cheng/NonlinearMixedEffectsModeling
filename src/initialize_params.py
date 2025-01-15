# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
################################################################## Initialization ######################################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// # 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def check_for_quadratic_relationship(x, y):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_standardized = (x - x_mean) / x_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    plt.scatter(x_standardized, y_standardized, color = 'blue', label = 'Data')

    # quadratic terms
    poly = PolynomialFeatures(degree = 2, include_bias = False)
    x_standardized = x_standardized.reshape(-1, 1)
    x_poly = poly.fit_transform(x_standardized) 
    
    # regression
    model = LinearRegression()
    model.fit(x_poly, y_standardized)
 
    x_min = np.min(x_standardized)
    x_max = np.max(x_standardized)

    x_fit = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    x_fit_poly = poly.transform(x_fit)
    y_fit = model.predict(x_fit_poly)  

    plt.plot(x_fit, y_fit, color = 'red', label='Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('E:\\check_for_quadratic.png', dpi = 900)
    plt.show()

    print("check for quadratic relationship (standarized):", f"y = {model.coef_[1]:.3f} * x^2 + {model.coef_[0]:.3f} * x + {model.intercept_:.3f}")
    
    # reduction of unstandardised coefficients
    coef_x2 = model.coef_[1] * (y_std / (x_std ** 2))
    coef_x = model.coef_[0] * (y_std / x_std)
    intercept = y_mean - coef_x * x_mean - coef_x2 * (x_mean**2)

    print("check for quadratic relationship (original):", f"y = {coef_x2:.3f} * x^2 + {coef_x:.3f} * x + {intercept:.3f}")


def nonlinear_model(X, a, b, c, d, e):
    x1, x2 = X
    return a * x1 ** 2 + b * x1 + c * x2 ** 2 + d * x2  + e


def initialize_params(x1, x2, y):
    x1_mean = np.mean(x1)
    x1_std = np.std(x1)
    x1_standardized = (x1 - x1_mean) / x1_std

    x2_mean = np.mean(x2)
    x2_std = np.std(x2)
    x2_standardized = (x2 - x2_mean) / x2_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std
    
    x1 = x1_standardized
    x2 = x2_standardized
    y = y_standardized

    x_data = np.vstack((x1_standardized.flatten(), x2_standardized.flatten()))
    popt, pcov = curve_fit(nonlinear_model, x_data, y.flatten(), p0 = [1, 1, 1, 1, 1])  # 初值

    y_pred = nonlinear_model(x_data, *popt)

    # visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # original points
    ax.scatter(x1, x2, y, color = 'blue', label = 'data')

    # predicts
    ax.scatter(x1, x2, y_pred, color = 'red', label = 'pred', alpha = 0.7)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title("Nonlinear Regression Fit")
    ax.legend()
    plt.show()

    return popt
