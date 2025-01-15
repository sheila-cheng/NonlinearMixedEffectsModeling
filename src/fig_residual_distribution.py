# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
######################################################## Residual Distributuion ########################################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// # 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def residual_distribution(df, trace):
    # predictions
    y_hat = trace.posterior["y_hat"].values
    y_hat = y_hat.reshape(-1, y_hat.shape[-1]) 
    predictions = y_hat * df['response_std'][0] + df['response_mean'][0]

    residual_samples = df['response'].to_numpy()[None, :] - predictions  # shape (8000, 162)
    var_residual = np.var(residual_samples, axis = 0)  # shape (162,)

    # print("First 5 residual samples:", residual_samples[:5, :5])
    
    # 可视化残差分布：分析残差是否呈现随机分布，且均值接近 0。
    sns.histplot(residual_samples.flatten(), kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    plt.savefig('E:\\residual_distribution.png', dpi = 900)
    plt.show()
