# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
############################################################# Model Structure ##########################################################
# off-nadir angle and no. of side-looking images are covariates, and 3D reconstruction accuracy is the dependent variable.

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
import pymc as pm
import numpy as np
import pytensor.tensor as pt

def build_model(df, _beta_quantity_squared, _beta_quantity, _beta_angle_squared, _beta_angle, _alpha, random_effects):
    with pm.Model() as model:
        # initialization
        alpha = pm.Normal('alpha', mu = _alpha, sigma = 0.1)  

        beta_quantity_squared = pm.Normal('beta_quantity_squared', mu = _beta_quantity_squared, sigma = 1.0) 
        beta_quantity = pm.Normal('beta_quantity', mu = _beta_quantity, sigma = 1.0)
        
        beta_angle_squared = pm.Normal('beta_angle_squared', mu = _beta_angle_squared, sigma = 1.0)
        beta_angle = pm.Normal('beta_angle', mu = _beta_angle, sigma = 1.0)
        
        n_combinations = np.unique(df['group'])

        # discrete
        group_effect = pm.Normal('group_effect', mu = 0, sigma = 1, shape = len(n_combinations))         
        mu = group_effect[df['group']] + alpha + beta_quantity * df['quantity'] + beta_angle * df['angle'] + beta_angle_squared * df['angle'] ** 2

        # residual for all observations
        sigma = pm.HalfNormal('sigma', sigma = 1)

        # likelihood function: normal distribution
        y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = df['response'])

        # ideal for quickly calculating predictions based on existing posterior distributions, for scenarios that do not require resampling
        y_hat = pm.Deterministic("y_hat", mu) 

        # iterately compute posterior distribution for model parameters: MCMC sample and iteration
        trace = pm.sample(2000, tune = 1000, target_accept = 0.95, nuts = {"max_treedepth": 15}, return_inferencedata = True)

    return trace