# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
################################################## Evaluation of Fitting Results #######################################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def compute_r_square(df, trace):
    # predictions
    y_hat = trace.posterior["y_hat"].values
    y_hat = y_hat.reshape(-1, y_hat.shape[-1]) 
    predictions = y_hat * df['response_std'][0] + df['response_mean'][0]

    n_groups = df['group'].nunique()
    predictions_grouped = np.zeros((predictions.shape[0], n_groups))
    observations_grouped = np.zeros(n_groups)
 
    for group_id in range(n_groups):
        mask = df['group'] == group_id
        predictions_grouped[:, group_id] = predictions[:, mask.values].mean(axis = 1)
        observations_grouped[group_id] = df.loc[mask.values, 'response'].mean()

    # calculate the explained variance and residual variance of the posterior distribution
    r2_posterior = np.zeros(predictions.shape[0])  

    for i in range(predictions.shape[0]):
        pred = predictions_grouped[i, :] 
        var_pred = np.var(pred)
        var_residual = np.var(predictions_grouped - pred)
        r2_posterior[i] = var_pred / (var_pred + var_residual)

    r2_mean = r2_posterior.mean()
    r2_hdi = np.percentile(r2_posterior, [2.5, 97.5])

    print(f"Posterior Bayesian R2 mean: {r2_mean:.3f}, 95% HDI: {r2_hdi:.3f}")
    # Posterior Bayesian R2 mean: 0.680, 95% HDI: [0.597 0.739]

    # draw plot
    sns.set(style = "whitegrid")
    plt.figure(figsize = (8, 6))
    
    r2_posterior_flat = r2_posterior.flatten()
    sns.kdeplot(r2_posterior_flat, fill=True, color="skyblue", alpha=0.7)
    
    plt.axvline(x = r2_mean, color = "red", linestyle = "--", label = f"Mean: {r2_mean:.3f}")
    plt.axvline(x = r2_hdi[0], color = "green", linestyle = "--", label = f"2.5%: {r2_hdi[0]:.3f}")
    plt.axvline(x = r2_hdi[1], color = "green", linestyle = "--", label = f"97.5%: {r2_hdi[1]:.3f}")

    plt.title("Posterior Distribution of Bayesian $R^2$", fontsize = 16)
    plt.xlabel("$R^2$", fontsize = 14)
    plt.ylabel("Density", fontsize = 14)
    plt.legend(fontsize = 12)
    plt.savefig('E:\\bayesian_r_square.png', dpi = 900)
    plt.show()