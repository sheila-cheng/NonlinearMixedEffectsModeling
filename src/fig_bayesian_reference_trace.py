# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
    
################################################## Plotting the Convergence Process ####################################################

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #

import arviz as az
import matplotlib.pyplot as plt

def bayesian_reference_trace(df, trace):
    trace_plot = az.plot_trace(trace, var_names=["alpha", "beta_quantity", "beta_quantity_squared", "beta_angle", "beta_angle_squared"]) 

    custom_colors = ["red",  "green", "#ff7f00","blue"]
    linestyles = ["--", "-.", ":", "-"]

    axes = plt.gcf().get_axes()  
    for ax in axes: 
        for j, line in enumerate(ax.get_lines()):
            line.set_color(custom_colors[j % len(custom_colors)])
            line.set_linestyle(linestyles[j % len(linestyles)])

    plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
    plt.savefig('E:\\trace_plot.png', dpi = 500)
    plt.show()