import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# set plotting parameters 
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
    'axes.labelsize': 18, # fontsize for x and y labels (was 10)
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
}
matplotlib.rcParams.update(params)

def add_annotate(ax):
    letter_list = [str(chr(k+97)) for k in range(0, 20)]
    k=0
    for i_ax, gca in enumerate(ax.flatten()):
        if i_ax !=6:
            gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes, fontsize=20)
            k+=1
    return ax

def plot_prediction_vs_measure_seaborn(peak_comparison_df, phase, bins=40, vmin=None, vmax=None, **kwargs):
    sns.set_theme(style="ticks", font_scale=2)
    if phase == 'P':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True, **kwargs)
                        #xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    elif phase == 'S':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True, **kwargs)
                        #xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    g.ax_joint.set(xscale="log")
    g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

# Add the joint and marginal histogram plots 03012d
    g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins*2,bins), cmap="light:#4D4D9C", vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", bins=bins, color="#4D4D9C")

    if 'xlim' in kwargs.keys():
        g.ax_joint.plot(kwargs['xlim'], kwargs['ylim'], 'k-', linewidth = 2)

    g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
    g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
    return g


def plot_magnitude_seaborn(df_magnitude, vmin=None, vmax=None, **kwargs): # TODO: think about the kwargs setting for plotting control
    sns.set_theme(style="ticks", font_scale=2)

    g = sns.JointGrid(data=df_magnitude, x="magnitude", y="predicted_M", marginal_ticks=True,**kwargs)
    #xlim=(1, 7), ylim=(1, 7), height=10, space=0.3)


    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

    # vmin, vmax = None, None
    # if 'vmin' in kwargs.keys():
    #     vmin = kwargs['vmin']
    # if 'vmax' in kwargs.keys():
    #     vmax = kwargs['vmax']

    # Add the joint and marginal histogram plots 03012d
    g.plot_joint(
    sns.histplot, discrete=(False, False), bins=(20, 20), vmax=vmax, vmin=vmin,
    cmap="dark:#fcd9bb_r", pmax=.7, cbar=True, cbar_ax=cax, cbar_kws={'label':'counts', 'spacing': 'proportional'})
    g.plot_marginals(sns.histplot, element="step", color="#c4a589") # light:#9C4D4D

    # ii_M4 = df_magnitude.magnitude >= 4 
    # g.ax_joint.plot(df_magnitude[ii_M4].magnitude, df_magnitude[ii_M4].predicted_M, 'k.', alpha=1)
    g.ax_joint.plot([-3,10], [-3,10], 'k-', linewidth = 2)
    g.ax_joint.plot([-3,10], [-2,11], 'k--', linewidth = 1)
    g.ax_joint.plot([-3,10], [-4,9], 'k--', linewidth = 1)
    g.ax_joint.set_xlabel('catalog magnitude')
    g.ax_joint.set_ylabel('predicted magnitude')

    if 'xlim' in kwargs.keys():
        g.ax_joint.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        g.ax_joint.set_ylim(kwargs['ylim'])

    return g

def plot_das_waveforms(strain_rate, das_time, gca, channel_index=None, title=None, pclip=None, **kwargs):

    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']

    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']

    if pclip:
        clipVal = np.percentile(np.absolute(strain_rate), pclip)
        vmin, vmax = -clipVal, clipVal
    
    if channel_index is None:
        clb = gca.imshow(strain_rate, 
                extent=[0, strain_rate.shape[1], das_time[-1],  das_time[0]],
                aspect='auto', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('seismic'), interpolation='none')
    else:
        clb = gca.imshow(strain_rate, 
                extent=[channel_index[0], channel_index[-1], das_time[-1],  das_time[0]],
                aspect='auto', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('seismic'), interpolation='none')


    gca.set_ylabel('Time (s)')
    gca.set_xlabel('channel number')
    
    if 'xmin' in kwargs.keys():
        gca.set_xlim(xmin=kwargs['xmin'])
    if 'xmax' in kwargs.keys():
        gca.set_xlim(xmax=kwargs['xmax'])
    if 'ymin' in kwargs.keys():
        gca.set_ylim(ymin=kwargs['ymin'])
    if 'ymax' in kwargs.keys():
        gca.set_ylim(ymax=kwargs['ymax'])

    axins1 = inset_axes(gca,
                        width="2%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        loc='lower right')
    if title:
        gca.set_title(title, fontsize=20)

    plt.colorbar(clb, cax=axins1, orientation="vertical", label='strain rate ($10^{-6}$/s)')
    return gca

def add_annotate(ax):
    letter_list = [str(chr(k+97)) for k in range(0, 20)]
    k=0
    for i_ax, gca in enumerate(ax.flatten()):
    #     gca.spines.right.set_visible(False)
    #     gca.spines.top.set_visible(False)
    # # add annotation
        gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes)
        k += 1
    return ax