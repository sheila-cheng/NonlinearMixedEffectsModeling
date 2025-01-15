# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
############################################################### Linearity Check ########################################################
############################# See if there is a linear relationship between predicted and observed values ##############################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// # 

import matplotlib.pyplot as plt

def linearity_check(df, trace):
    # predictions
    y_hat = trace.posterior["y_hat"].values
    y_hat = y_hat.reshape(-1, y_hat.shape[-1]) 
    mean_pred = y_hat.mean(axis = 0)
  
    # draw figure
    plt.figure(figsize = (12, 8))

    # for angle in df['angle'].unique():
    #     subset = df[df['angle'] == angle]
    #     indices = subset.index
    #     plt.scatter(
    #         subset['response'], 
    #         mean_pred[indices] * df['response_std'][0] + df['response_mean'][0],
    #         label = f'angle {angle}', 
    #         alpha = 0.7
    #     )

    for quantity in df['quantity'].unique():
        subset = df[df['quantity'] == quantity]
        indices = subset.index
        plt.scatter(
            subset['response'], 
            (mean_pred[indices] * df['response_std'][0] + df['response_mean'][0]),
            label = f'quantity {quantity}', 
            alpha = 0.7
        )

    plt.plot(
        [df['response'].min(), df['response'].max()], 
        [df['response'].min(), df['response'].max()], 
        'r--', label = 'Ideal Fit'
    )

    plt.xlabel('obserations')
    plt.ylabel('predictions')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig('E:\\linearity_check.png', dpi = 900)
    plt.show()

