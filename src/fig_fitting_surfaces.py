# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
############################################## Three-dimensional Diagrams (simulation + real data) #####################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

def fitting_surface_with_observations_and_predictions(df, trace, OMA_data, Luojia3_01_data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#0260A0',
              '#3CA4E5',
              '#38BDEA',
              '#93D2F5',
              '#D6DAF7']

    # draw observation dotts (synthetic data)
    for idx, quantity in enumerate(df['quantity'].unique()):
        subset = df[df['quantity'] == quantity]
        angles = subset['angle']
               
        ax.scatter(
            subset['quantity'], 
            subset['angle'], 
            subset['response'], 
            c = colors[idx],
            alpha = 0.8, 
            label = f'n = {quantity}', 
            # edgecolors = 'black', 
            # linewidth = 1, 
            s = 30, 
            marker = 'o'
        )

    # draw observation dotts (luojia3-01 data)
    ax.scatter(
        Luojia3_01_data['luojia3_01_quantity'],  # x: quantity
        Luojia3_01_data['luojia3_01_angle'],     # y: angle
        Luojia3_01_data['luojia3_01_response'],  # z: response
        color = 'green',          
        label = 'Luojia3_01',   
        s = 30                  
    )

    # draw observation dotts (US3D_OMA)
    ax.scatter(
        OMA_data['OMA_342_quantity'],  # x: quantity
        OMA_data['OMA_342_angle'],     # y: angle
        OMA_data['OMA_342_response'],  # z: response
        color = 'blue',      
        label = 'OMA_324',  
        s = 30                  
    )

    # predictions
    y_hat = trace.posterior["y_hat"].values
    y_hat = y_hat.reshape(-1, y_hat.shape[-1]) 
    predictions = y_hat * df['response_std'][0] + df['response_mean'][0]

    # directly predicts the posterior distribution for each observation record (including noises), 
    # not the posterior distribution for the unique combination of (angle, number).
    # To obtain a predictive distribution of unique groups (angle, number), the predicts can be aggregated or averaged:
    n_groups = df['group'].nunique()
    predictions_grouped = np.zeros((predictions.shape[0], n_groups))

    for group_id in range(n_groups):
        mask = df['group'] == group_id
        predictions_grouped[:, group_id] = predictions[:, mask.values].mean(axis = 1)

    lower_percentile = 2.5
    upper_percentile = 97.5
    mean_pred = np.mean(predictions_grouped, axis = 0)
    lower_pred = np.percentile(predictions_grouped, lower_percentile, axis = 0)
    upper_pred = np.percentile(predictions_grouped, upper_percentile, axis = 0)
    
    # draw prediction dotts and surfaces
    unique_quantities = df['quantity'].unique()
    unique_angles = df['angle'].unique()

    # quantity_grid, angle_grid = np.meshgrid(unique_quantities, unique_angles)
    # interp_lower = interpolate.interp2d(unique_quantities, unique_angles, lower_pred, kind='cubic')
    # interp_upper = interpolate.interp2d(unique_quantities, unique_angles, upper_pred, kind='cubic')
    # lower_surface = interp_lower(unique_quantities, unique_angles)
    # upper_surface = interp_upper(unique_quantities, unique_angles)

    indices, unique_combinations = pd.factorize(df[['quantity', 'angle']].apply(tuple, axis = 1))
    unique_combinations = np.array(unique_combinations.tolist(), dtype = int)

    quantity_grid, angle_grid = np.meshgrid(unique_quantities, unique_angles, indexing = 'ij')
    points_grid = np.column_stack([quantity_grid.ravel(), angle_grid.ravel()])
    interp_lower = interpolate.griddata(points = unique_combinations, values = lower_pred, xi = points_grid, method = 'cubic')
    interp_upper = interpolate.griddata(points = unique_combinations, values = upper_pred, xi = points_grid, method = 'cubic')
    
    lower_surface = interp_lower.reshape(quantity_grid.shape)
    upper_surface = interp_upper.reshape(quantity_grid.shape)
    
    ax.plot_surface(quantity_grid, angle_grid, lower_surface, cmap = 'OrRd', alpha = 0.3)
    ax.plot_surface(quantity_grid, angle_grid, upper_surface, cmap = 'OrRd', alpha = 0.3)
  
    # draw mean of prediction dotts
    for idx, (q, a) in enumerate(unique_combinations):
        ax.scatter(q, a, mean_pred[idx], 
           c = 'red',          
           edgecolors = 'black',  
           s = 60,               
           linewidth = 1,       
           marker = 'o')         
    
    ax.set_xticks(np.arange(2, 6.1, 1))   # x axis: quantity
    ax.set_yticks(np.arange(5, 40.1, 5))  # y axis: angle
    ax.set_zticks(np.arange(0, 5.1, 1))   # z axis: response
    
    ax.set_xlim([2, 6])  
    ax.set_ylim([5, 40])
    ax.set_zlim([0, 5])
  
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel('no. of side-looking images', fontsize = 14)
    ax.set_ylabel('off-nadir angle', fontsize = 14)
    ax.set_zlabel('3D city modeling accuracy', fontsize = 14)

    ax.legend(loc='upper left', fontsize=12)

    # background color
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.grid(False)
   
    ax.view_init(elev = 45, azim = 45)
    # ax.view_init(elev = 0, azim = 0)
    
    fig.savefig('E://3d_fitting_surface_with_observations_and_predictions.png', dpi = 900)

    plt.tight_layout()
    plt.show()

