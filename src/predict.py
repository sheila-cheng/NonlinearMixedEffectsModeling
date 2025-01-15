import fig_fitting_surfaces
import fig_linearity_check
import fig_residual_distribution
import fig_bayesian_r_square
import fig_bayesian_reference_trace

import initialize_params
import build_mixed_effect_model

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from scipy import interpolate
import pytensor.tensor as pt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
######################################################### Hyper-Parameters #############################################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

if __name__ == '__main__':
    
    # synthetic data
    # image resolution: 0.5m
    synthetic_data = {
        'quantity': [   
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
          
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3,

            4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 
            4, 4, 4, 4, 4, 4, 4, 4, 4, 

            5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5,

            6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6
        ],

        'angle': [
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 

            5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 

            5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 

            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 
            5, 10, 15, 20, 25, 30, 35, 40, 45, 

            5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        ],
        
        'response': [
            0.925104, 0.696525, 0.673074, 0.619206, 0.591175, 0.561352, 0.639062, 0.715684, 0.90088, 1.01306,
            1.06293, 0.718953, 0.609944, 0.655953, 0.600646, 0.62405, 0.765502, 0.872201, 0.78743,
            1.15661, 0.714976, 0.755306, 0.621188, 0.559399, 0.878308, 0.678127, 0.816049, 0.974781,
            0.94109, 0.63742, 0.608083, 0.540218, 0.5614, 0.584883, 0.591806, 0.865128, 0.973785,
            1.25031, 0.663527, 0.724428, 0.54172, 0.573604, 0.823638, 0.769236, 0.761974, 3.93528,
            1.30329, 0.702715, 0.603924, 0.657752, 0.567961, 0.610556, 0.718803, 0.69776, 3.84452,

            0.784666, 0.63933, 0.738003, 0.592133, 0.669037, 0.691152, 1.08871, 0.924238, 1.17766,0.8707765,
            1.07755, 0.692611, 0.563459, 0.682633, 0.653355, 0.651609, 0.773263, 0.583412, 3.91142,
            0.880475, 0.658675, 0.766909, 0.53917, 0.726967, 0.690785, 0.541967, 0.604458, 0.670676,
            1.02067, 0.651394, 0.86579, 0.544225, 0.719192, 0.781039,2.29903, 0.660998, 1.41824,

            0.745975, 0.574221, 0.55669, 0.525594, 0.701314, 0.526364, 0.624775, 0.684105, 0.75309, 0.856773,
            0.986737, 0.687571, 0.66078, 0.583917, 0.52642, 0.957623, 1.07207, 1.02964, 0.933544,
            0.973421, 0.676303, 0.648467, 0.561411, 0.561145, 0.789196,0.741074, 0.763976, 1.05705,
            0.961875, 0.827849, 0.726138, 0.602761, 0.543073, 0.705717,1.0902, 0.727877, 0.625722,

            1.08885, 0.692532, 0.576366,0.550919,0.532262,0.800353,0.783001,0.845095,0.968022,
            0.819996, 0.706251, 0.556472,0.515404,0.943301,0.773962,0.645822,0.870046,0.932055,

            0.937845, 0.732733, 0.753987, 0.531173, 0.649325, 0.76423, 1.30473, 1.16837, 0.936586, 1.02266,
            1.0085, 0.745454, 0.619002, 0.647138, 0.607489, 0.732101, 1.40634, 0.862921, 1.02534, 2.13132
        ],
    }

    # US3D data, OMA_342
    # no. of side-looking images = 4, with varying angles    
    # image resolution: 0.35m
    OMA_data = {    
        'OMA_342_quantity' : [
            # 11-17 5 组
            4,4,4,4,4,
            # 18-22 35 组
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,
            # 23-29 43 组
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,4,4,4,4,4,4,4,
            4,4,4,
        ],

        'OMA_342_angle' : [
            # 11-17 5 组
            14,14,14,14,14,
            # 18-22 35 组
            20,20,20,20,20,20,20,20,20,20,
            20,20,20,20,20,20,20,20,20,20,
            20,20,20,20,20,20,20,20,20,20,
            20,20,20,20,20,
            # 23-29 43 组
            26,26,26,26,26,26,26,26,26,26,
            26,26,26,26,26,26,26,26,26,26,
            26,26,26,26,26,26,26,26,26,26,
            26,26,26,26,26,26,26,26,26,26,
            26,26,26,
        ],

        'OMA_342_response' : [
            # 11-17 5 组
            0.961593287,1.090387349,1.670880585,0.935151856,1.134993898,
            # 18-22 35 组
            0.837736826,0.915539896,0.92919841,0.958972355,0.964392659,0.971241804,0.980278986,1.014122611,1.016089344,1.023071397,
            1.042958887,1.048477478,1.066319566,1.074938765,1.089107006,1.132790949,1.133072291,1.133395019,1.149457349,1.155080865,
            1.178138783,1.190529933,1.213883978,1.227926527,1.23254263,1.257243437,1.257947284,1.311020537,1.380661317,1.42351694,
            1.464487068,1.656548431,1.716838355,1.752968789,2.016650159,
            # 23-29 43 组
            0.90516821,1.418539928,0.855454818,1.394118115,1.352909407,1.200106249,1.110092967,1.130060514,1.384568852,1.299577282,
            1.011302785,1.06284571,1.057900636,1.499612178,3.125904227,0.706679076,1.340660486,0.776991138,1.475182127,0.952733657,
            1.590434783,1.444568086,1.045847332,1.06029825,0.779543722,1.290539556,1.156302384,0.936035495,1.255031946,0.991015111,
            1.571915655,0.749536082,0.987603525,1.170907517,1.29828821,1.535724269,1.374998721,1.017869604,1.091804018,1.210038937,
            1.047475093,1.108508969,1.178305816
        ],
    }

    # Luojia3-01 satellite video data, 
    # no. of side-looking images = 2, with the off-nadir angle of 11.53717381° (the angle between the first and last images as 23.07434762°）
    # the resolution is not taken into account, because the number of offset pixels on the image is measured directly.
    Luojia3_01_data = {
        'luojia3_01_quantity' : [
            2,2,2,2,2,2,2,2,2,2,
            2,2,2,2,2,2,2,2,2,2,
            2,2,2,2,2,2,2,2,2,
        ],

        'luojia3_01_angle' : [
            11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 
            11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 
            11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 11.54, 
        ],
        
        'luojia3_01_response' : [
            # 29 组
            2.654127165,2.571189379,2.474184498,2.339782028,2.381753124,2.49404376,2.590120091,2.342533353,2.38201527,2.410470729,
            2.388934734,2.338408018,2.307986682,2.356795196,2.444099718,2.286824444,2.345733328,2.405086807,2.35072278,2.375126043,
            2.382121835,2.33996673,2.350210523,2.336109175,2.345837427,2.324465227,2.377142088,2.297114468,2.291695076
        ],   
    }
    
    # times of ground sample distance
    image_resolution = 0.5
    synthetic_data['response'] = [x / image_resolution for x in synthetic_data['response']]

    image_resolution = 0.35
    OMA_data['OMA_342_response'] = [x / image_resolution for x in OMA_data['OMA_342_response']]

    df = pd.DataFrame(synthetic_data)
    # exclude rows with angles 45°, 50°, 
    # as we found that the variance of the data in this range was too large and no longer met the assumptions of the quadratic relationship
    df = df[~df['angle'].isin([45, 50])]
    df = df.reset_index(drop = True)

    # preprocessing: normalization
    df['angle_mean'] = np.mean(df['angle'])
    df['quantity_mean'] = np.mean(df['quantity'])
    df['response_mean'] = np.mean(df['response'])

    df['angle_std'] = np.std(df['angle'])
    df['quantity_std'] = np.std(df['quantity'])
    df['response_std'] = np.std(df['response'])

    df_normalized = df.copy()

    quantity_standardized = (df['quantity'] - df['quantity_mean']) / df['quantity_std']
    df_normalized['quantity'] = quantity_standardized

    angle_standardized = (df['angle']- df['angle_mean']) / df['angle_std']
    df_normalized['angle'] = angle_standardized

    response_standardized = (df['response'] - df['response_mean']) / df['response_std']
    df_normalized['response'] = response_standardized

    # add a new column 'group' to assign a unique number to each (quantity, angle) combination
    indices, unique_combinations = pd.factorize(df[['quantity', 'angle']].apply(tuple, axis = 1))
    df['group'] = indices
    df_normalized['group'] = indices

    n_groups = np.unique(df['group'])

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
    ######################################################### Quadric Check ################################################################

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    # # supplementary figure 1: Quadric relationship between parameters and accuracy for initializing fixed effects in NLME model.
    # initialize_params.check_for_quadratic_relationship(synthetic_data['angle'], synthetic_data['response'])
    # check for quadratic relationship (standarized): y = 0.419 * x^2 + 0.257 * x + -0.419
    # check for quadratic relationship (original): y = 0.002 * x^2 + 0.018 * x + -0.289

    # initialize_params.check_for_quadratic_relationship(synthetic_data['quantity'], synthetic_data['response'])
    # check for quadratic relationship (standarized): y = 0.102 * x^2 + -0.068 * x + -0.102
    # check for quadratic relationship (original): y = 0.050 * x^2 + -0.047 * x + 1.215
    
    initialize_params.check_for_quadratic_relationship(df['quantity'].to_numpy(), df['response'].to_numpy())
    # check for quadratic relationship (standarized): y = 0.080 * x^2 + 0.066 * x + -0.080
    # check for quadratic relationship (original): y = 0.019 * x^2 + 0.021 * x + 1.200

    initialize_params.check_for_quadratic_relationship(df['angle'].to_numpy(), df['response'].to_numpy())
    # check for quadratic relationship (standarized): y = 0.562 * x^2 + -0.001 * x + -0.562
    # check for quadratic relationship (original): y = 0.002 * x^2 + -0.000 * x + 0.541

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
    ############################################################# Training ################################################################

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    # initialization
    a, b, c, d, e = initialize_params.initialize_params(df['quantity'].to_numpy(), df['angle'].to_numpy(), df['response'].to_numpy())
    print("initialized parameters:", f"beta_q_2 = {a:.3f}, beta_q = {b:.3f}, beta_a_2 = {c:.3f}, beta_a = {d:.3f}, alpha = {e:.3f}")
    # initialized parameters: beta_q_2 = 0.080, beta_q = 0.066, beta_a_2 = 0.562, beta_a = -0.001, alpha = -0.643
    
    # mixed-effect model
    trace = build_mixed_effect_model.build_model(df_normalized, a, b, c, d, e)
     
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    ########################################################### related figures ############################################################

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    # fig.6 3D visualization of a trained nonlinear mixed-effect model for predicting 3d city modeling accuracy using synthetic data
    fig_fitting_surfaces.fitting_surface_with_observations_and_predictions(df, trace, OMA_data, Luojia3_01_data)

    # supplementary figure 2: Details for nonlinear mixed-effect model itting
    # linearity check
    fig_linearity_check.linearity_check(df, trace)
    
    # residual distribution
    fig_residual_distribution.residual_distribution(df, trace)

    # trace plot
    fig_bayesian_reference_trace.bayesian_reference_trace(df, trace)

    # bayesian r2
    fig_bayesian_r_square.compute_r_square(df, trace)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

    ####################################################### posterior distribution #########################################################

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
 
    # shape (4, 2000)
    beta_quantity = trace.posterior['beta_quantity'].to_numpy()    
    beta_quantity_squared = trace.posterior['beta_quantity_squared'].to_numpy()
    beta_angle = trace.posterior['beta_angle'].to_numpy()
    beta_angle_squared = trace.posterior['beta_angle_squared'].to_numpy()
    alpha = trace.posterior['alpha'].to_numpy()
    sigma = trace.posterior['sigma'].to_numpy() 

    group_effect = trace.posterior['group_effect'].to_numpy() # shape (4, 2000, n_groups)
    y_hat= trace.posterior["y_hat"].values  # shape (4, 2000, n_observations)
    

    # parameters of fixed effect (mean)
    print("fitting parameters:", f"beta_q_2 = {np.mean(beta_quantity_squared)}, beta_q = {np.mean(beta_quantity)}, \
          beta_a_2 = {np.mean(beta_angle_squared)}, beta_a = {np.mean(beta_angle)}, \
          alpha = {np.mean(alpha)}, sigma = {np.mean(sigma)}")
    # fitting parameters: beta_q_2 = 0.09927527984969259, beta_q = 0.0548746440696786,
    # beta_a_2 = 0.5589417083693968, beta_a = 0.06250703609111041,
    # alpha = -0.6445036327391072, sigma = 0.7663819087104395

    # parameters of random effect: shape（n_groups,)
    for idx,(q,a) in enumerate(unique_combinations):
        print(f"{idx}, ({q},{a}), {group_effect[:,:, idx].mean()}")
    
    # extreme point of the fitted model
    predictions = y_hat.reshape(-1, y_hat.shape[-1]) 
    predictions = predictions * df['response_std'][0] + df['response_mean'][0]

    min_value = np.min(predictions)
    min_index = np.argmin(predictions)

    selected_row = df[df['group'] == min_index]

    min_quantity = selected_row['quantity'].iloc[0]
    min_angle = selected_row['angle'].iloc[0]
    
    print(f"minimum point of the fitted model: quality = {min_value}, quantity = {min_quantity}, angle = {min_angle}")
    # minimum point of the fitted model: quantity = 1.13945512002938, quantity = 4, angle = 20


    # for Luojia3-01 video data, predicting the reconstruction quality under the parameter (2, 11.53717381)
    pre_quantity = 2
    pre_angle = 11.53717381
    unique_quantities = df['quantity'].unique()
    unique_angles = df['angle'].unique()
   
    interp_func = interpolate.interp2d(unique_quantities, unique_angles, predictions.mean(axis=0), kind = 'linear')
    print(f"single point prediction: {interp_func(2, 11.53717381)}") 
    # single point prediction: 1.40151591
    
   