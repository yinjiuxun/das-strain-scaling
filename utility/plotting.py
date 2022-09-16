import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase, bins=40, vmin=None, vmax=None):
    sns.set_theme(style="ticks", font_scale=2)
    if phase == 'P':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    elif phase == 'S':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    g.ax_joint.set(xscale="log")
    g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

# Add the joint and marginal histogram plots 03012d
    g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins*2,bins), cmap="light:#4D4D9C", vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", bins=bins, color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
    g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
    return g