import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
import seaborn as sns
import scipy
from scipy.signal import sawtooth
from scipy.stats import linregress
from src.dataproc import utils
from astropy.timeseries import LombScargle
from astropy.stats.circstats import circmean, circvar



def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, facecolor='tab:blue'):
    """
    Produce a circular histogram of angles on ax.

    Code taken from Stackexchange: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
    on 12th April 2024.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='black', fill=True, facecolor=facecolor, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def create_jointplot_figure(x, y, z, gridsize=20, jointcmap='Blues', textfontsize=11, scregparams=None):
    """
    Creates a figure with three subplots similar to seaborn.jointplot's layout.

    The first subplot is a scatterplot of x vs. z with a regression line.
    The second subplot is a scatterplot of y vs. z with a regression line.
    The third subplot is a hexbin plot of x vs. y with z as the weight.

    Args:
        x (np.array): The first continuous variable.
        y (np.array): The second continuous variable.
        z (np.array): The third continuous variable (weight for hexbin).
    """
    # get color for scatterplot
    fig_cmap = plt.get_cmap(jointcmap)
    central_color_rgba = fig_cmap(0.5) # Get the color at the center (0.5)

    # --- 1. Set up the figure and subplots ---
    # We use a 2x2 grid to create the desired jointplot-like layout.
    # The first two plots (x-z and y-z) will be on the diagonal, and the
    # hexbin plot will be in the main central area.
    # We use gridspec to control the placement and relative sizes of the subplots.
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3,  # 2 rows, 2 columns
                          width_ratios=[0.15, 4, 1],  # Main plot (x-z) takes more width
                          height_ratios=[1, 4], # Main plot (x-z) takes more height
                          wspace=0.05, hspace=0.05)

    # --- 2. Create the individual subplot axes ---
    ax_cbar = fig.add_subplot(gs[1, 0])
    # The main hexbin plot is in the large bottom-left space.
    ax_hexbin = fig.add_subplot(gs[1, 1])
    # The top-left corner is for the x-z scatter plot.
    ax_xz_scatter = fig.add_subplot(gs[0, 1], sharex=ax_hexbin)
    # The bottom-right corner is for the y-z scatter plot.
    ax_yz_scatter = fig.add_subplot(gs[1, 2], sharey=ax_hexbin)

    # --- 3. Create the hexbin plot with z as weight ---
    # The hexbin plot shows the relationship between x and y, with color
    # representing the average value of z in each hexbin.
    hb = ax_hexbin.hexbin(x, y, C=z, gridsize=gridsize, cmap=jointcmap)
    ax_hexbin.set_xlabel('x', fontsize=12)
    ax_hexbin.set_ylabel('y', fontsize=12)

    # Add the colorbar, placing it in the dedicated `ax_cbar` axes
    cbar = fig.colorbar(hb, cax=ax_cbar, shrink=0.5)
    cbar.set_label('Average z value', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # --- 4. Create the scatterplot of x vs. z ---
    # The scatter plot shows the relationship between x and z.
    ax_xz_scatter.scatter(x, z, s=20, alpha=0.6, color=central_color_rgba)
    ax_xz_scatter.set_title('x vs. z', fontsize=14)
    # Fit and plot a linear regression line to show the trend.
    if scregparams is None:
        slope, intercept, r_value, p_value, std_err = linregress(x, z)
    else:
        slope = scregparams['xz']['slope']
        intercept = scregparams['xz']['intercept']
        r_value = scregparams['xz']['r_value']
        p_value = scregparams['xz']['p_value']

    x_line = np.array([min(x), max(x)])
    y_line = slope * x_line + intercept
    ax_xz_scatter.plot(x_line, y_line, color='black', linestyle='--')
    ax_xz_scatter.text(x=0.9, y=0.2, s='$r=%0.3f$\n$p=%0.4f$'%(r_value, p_value), 
        transform=ax_xz_scatter.transAxes, ha='center', fontsize=textfontsize)
    ax_xz_scatter.set_ylabel('z', fontsize=12)
    # Hide the x-axis labels on the top plot to make the figure cleaner.
    ax_xz_scatter.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    # ax_xz_scatter.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Create the scatterplot of y vs. z ---
    # The scatter plot shows the relationship between y and z.
    ax_yz_scatter.scatter(z, y, s=20, alpha=0.6, color=central_color_rgba)
    ax_yz_scatter.set_title('y vs. z', fontsize=14)
    # Fit and plot a linear regression line.
    if scregparams is None:
        slope, intercept, r_value, p_value, std_err = linregress(y, z)
    else:
        slope = scregparams['yz']['slope']
        intercept = scregparams['yz']['intercept']
        r_value = scregparams['yz']['r_value']
        p_value = scregparams['yz']['p_value']
    
    y_line = np.array([min(y), max(y)])
    z_line = slope * y_line + intercept
    ax_yz_scatter.plot(z_line, y_line, color='black', linestyle='--')
    ax_yz_scatter.text(x=0.7, y=0.85, s='$r=%0.3f$\n$p=%0.4f$'%(r_value, p_value), 
        transform=ax_yz_scatter.transAxes, ha='center', fontsize=textfontsize)
    ax_yz_scatter.set_xlabel('z', fontsize=12)
    # Hide the y-axis labels on the right plot for a cleaner look.
    ax_yz_scatter.tick_params(axis='y', which='both', left=True, labelleft=False)
    # ax_yz_scatter.grid(True, linestyle='--', alpha=0.6)

    return fig, ax_hexbin, ax_xz_scatter, ax_yz_scatter, ax_cbar



def LSpointplot(ax, subfeatdf, ymean, yerr, featname, color, marker='o', leglabel=''):
    # error bar plot
    ax.errorbar(x=ymean['Round_local_hour'], y=ymean[featname], 
                yerr=yerr[featname], linestyle='none', elinewidth=2, 
                marker='o', markersize=8, color=color)
    
    # Lomb Scarlge fit just for visualization
    vis_lsp_object = LombScargle(t=subfeatdf['Round_local_hour'].values, 
                                 y=subfeatdf[featname].values, 
                                 fit_mean=True)
    vis_lsp_offset = utils.lsmodel_phase_offset(vis_lsp_object, frequency=1/24) # phase offset of the LS model (in radians)
    vis_peak_angle = utils.lsmodel_phase_offset_peak(vis_lsp_offset) # angle at which LS model reaches highest value (sin (angle) = 1) (in radians)
    vis_ls_amp = np.linalg.norm(vis_lsp_object.model_parameters(frequency=1/24)[1:])
    vis_ls_pow = vis_lsp_object.power(frequency=1/24)
    vis_ls_pval = vis_lsp_object.false_alarm_probability(vis_ls_pow)

    # lomb scargle plot
    ax.plot(np.arange(0,24), vis_lsp_object.model(t=np.arange(0,24), frequency=1/24), 
                 linewidth=2, color=color, label=leglabel)  # color the model based on its significance
    
    # peak hour based on Lomb Scargle
    vis_peak_hour = 24*vis_peak_angle/(2*np.pi)
    ax.axvline(x=vis_peak_hour, color=color, linestyle='--')
    plot_ylim = ax.get_ylim()
    ax.scatter(vis_peak_hour, plot_ylim[1], s=200, c=color, marker=marker, 
        edgecolor='k', clip_on=False)
    ax.set_ylim(plot_ylim)

    return vis_peak_hour



def LSpointplot_polar(ax, subfeatdf, ymean, yerr, featname, color, show_points=True, show_lsamp=True, marker='o', linewidth=1):
    '''Polar plot, so all hours are converted to angles.'''

    if show_points:
        # error bar plot
        ax.errorbar(x=np.pi/12 * ymean['Round_local_hour'], y=ymean[featname],
                    yerr=yerr[featname], linestyle='none', elinewidth=2, 
                    marker='o', markersize=8, color=color)
    
    
    # Lomb Scarlge fit just for visualization
    vis_lsp_object = LombScargle(t=subfeatdf['Round_local_hour'].values, 
                                 y=subfeatdf[featname].values, 
                                 fit_mean=True)
    vis_lsp_offset = utils.lsmodel_phase_offset(vis_lsp_object, frequency=1/24) # phase offset of the LS model (in radians)
    vis_peak_angle = utils.lsmodel_phase_offset_peak(vis_lsp_offset) # angle at which LS model reaches highest value (sin (angle) = 1) (in radians)
    vis_ls_amp = np.linalg.norm(vis_lsp_object.model_parameters(frequency=1/24)[1:])
    vis_ls_pow = vis_lsp_object.power(frequency=1/24)
    vis_ls_pval = vis_lsp_object.false_alarm_probability(vis_ls_pow)

    if show_lsamp:
        # lomb scargle plot
        ax.plot(np.pi / 12 * np.arange(0,25), vis_lsp_object.model(t=np.arange(0,25), frequency=1/24), 
                     linewidth=1, color=color)  # color the model based on its significance
        
    # peak hour based on Lomb Scargle
    ax.axvline(x=vis_peak_angle, color=color, linestyle='--', linewidth=linewidth)
    plot_rmin, plot_rmax = ax.get_rmin(), ax.get_rmax()
    ax.scatter(vis_peak_angle, plot_rmax, s=500, c=color, marker=marker, 
        edgecolor='k', clip_on=False)
    ax.set_rlim([plot_rmin, plot_rmax])

    return vis_peak_angle



def lightning_bolt():
    # Define the vertices and drawing commands for the lightning bolt shape
    # These coordinates are centered around (0,0) and normalized for Matplotlib.

    verts = [
       ( 0.4   ,  1.6),
       ( 1.28  ,  0.96),
       ( 0.48  ,  0.2144),
       ( 0.736 ,  0.),
       ( 0.    , -0.64  ),
       ( 0.24  , -0.832 ),
       (-0.96  , -1.92  ),
       (-0.24  , -0.48  ),
       (-0.416 , -0.32  ),
       ( 0.128 ,  0.512 ),
       (-0.032 ,  0.64  ),
       ( 0.4   ,  1.6   )
       ]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    # Create the Path object for the marker
    lightning_bolt = Path(verts, codes)
    return lightning_bolt


def arrow_arc(ax, start_angle, arrow_radius, arrow_length, direction='clockwise', color='k'):

    if direction == 'clockwise':
        ax.plot(np.linspace(start_angle + arrow_length, start_angle + 2*arrow_length, 10), arrow_radius * np.ones(10), color=color)
        ax.arrow(x=start_angle + 2*arrow_length, y=arrow_radius, dx=arrow_length, dy=0, head_starts_at_zero=False,
                      length_includes_head=True, head_width=0.1, color=color)

    elif direction == 'anticlockwise':
        ax.plot(np.linspace(start_angle - arrow_length, start_angle - 2*arrow_length, 10), arrow_radius * np.ones(10), color=color)
        ax.arrow(x=start_angle - 2*arrow_length, y=arrow_radius, dx=-arrow_length, dy=0, head_starts_at_zero=False,
                      length_includes_head=True, head_width=0.1, color=color)
        
    return None


def outer_arrow_arc(ax, start_angle, arrow_radius, arrow_length, arc_curve=1, direction='clockwise', color='k'):
    
    if direction == 'clockwise':

        left_arrow = mpatches.FancyArrowPatch(
            (start_angle, arrow_radius),
            (start_angle - arrow_length, arrow_radius),
            connectionstyle=f'arc3,rad={arc_curve*np.pi/30}',
            arrowstyle='-|>',
            mutation_scale=20,
            color=color,
            clip_on=False,
            zorder=2
        )
        ax.add_patch(left_arrow)

    elif direction == 'anticlockwise':

        right_arrow = mpatches.FancyArrowPatch(
            (start_angle, arrow_radius),
            (start_angle + arrow_length, arrow_radius),
            connectionstyle=f'arc3,rad={-arc_curve*np.pi/30}',
            arrowstyle='-|>',
            mutation_scale=20,
            color=color,
            clip_on=False,
            zorder=2
        )
        ax.add_patch(right_arrow)

    return None

def arrow_arc_long(ax, start_angle, end_angle, arrow_radius, color='k', linestyle='-', arrowend='both'):
    
    head_sign = np.sign(end_angle - start_angle)

    ax.plot(np.linspace(start_angle, end_angle, 30), arrow_radius * np.ones(30), linewidth=2, 
        linestyle=linestyle, color=color)

    # arrow heads
    if arrowend in ['start', 'both']:
        ax.arrow(x=start_angle + np.pi/360, y=arrow_radius, dx= -1*head_sign*np.pi/360, dy=0, head_starts_at_zero=False,
                 length_includes_head=True, head_width=0.1, color=color)

    if arrowend in ['end', 'both']:
        ax.arrow(x=end_angle - np.pi/360, y=arrow_radius, dx= head_sign*np.pi/360, dy=0, head_starts_at_zero=False,
            length_includes_head=True, head_width=0.1, color=color)

    return None


def kuramoto_forward_trajectories(osppdf, hue_list, hue_colors, theta_left, theta_right, 
    fig=None, ax=None, legend=True, markersize=500):

    if (ax is None):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    rmax = osppdf['peaknum'].max()
    
    for oscname in hue_list:
        ax.plot(osppdf.loc[osppdf['osc']==oscname]['peakphase'],
                osppdf.loc[osppdf['osc']==oscname]['peaknum'], color=hue_colors[oscname], 
                linewidth=2, linestyle='-', label=oscname)
        # if oscname in ['left', 'right']:
        #     ax.scatter(osppdf.loc[ osppdf.loc[osppdf['osc']==oscname]['peaknum'].idxmax() ]['peakphase'], rmax, 
        #                s=2*markersize, c=hue_colors[oscname], marker='*', clip_on=False)    

    # plot the ground truth hour
    ax.scatter(theta_left, rmax, s=markersize, c='tab:blue', marker='^', clip_on=False)
    ax.scatter(theta_right, rmax, s=markersize, c='tab:orange', marker='^', clip_on=False)

    ax.set_ylim([0, rmax])

    # aesthetics
    ax.axvspan(xmin=(21-24) * np.pi/12, 
               xmax=9 * np.pi/12, color='grey', alpha=0.1)

    ax.set_theta_zero_location("N")  # Set 0 at the top
    ax.set_theta_direction(-1)      # Clockwise direction
    ax.set_xticks(ticks=np.arange(0,8)*np.pi/4, 
                  labels=['12AM', '3AM', '6AM', '9AM', '12PM', '3PM', '6PM', '9PM'])

    if legend: ax.legend()

    return fig, ax


def plot_cartesian_spring_chord_zigzag(
    ax, theta_start, theta_end, r=1.0, 
    coil_amplitude=0.015, wavelength=0.05, 
    num_coils=4, color='darkgreen'
):
    """
    Plots a sharp, zigzag spring pattern along a straight chord (Cartesian) 
    connecting two points on the circumference of a polar plot.

    Args:
        ax (matplotlib.axes.Axes): The polar axes object to plot on.
        theta_start (float): The starting angle (radians).
        theta_end (float): The ending angle (radians).
        r (float): The radius (distance from center) of the markers.
        coil_amplitude (float): The maximum height of the coil wave relative to the chord.
        wavelength (float): The desired horizontal distance (in Cartesian units) 
                            between the peaks and troughs of the zigzag.
        straight_ratio (float): Fraction of total length to use for straight ends (0.0 to 0.5).
    """
    
    # 1. Convert marker polar coordinates (r, theta) to Cartesian (x, y)
    x1 = r * np.cos(theta_start)
    y1 = r * np.sin(theta_start)
    x2 = r * np.cos(theta_end)
    y2 = r * np.sin(theta_end)
    
    # Calculate total length of the chord
    chord_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if chord_length == 0:
        return # Cannot draw a spring between the same point

    # Define the number of full coils based on length and desired wavelength
    num_coils_maximum = int(np.ceil(chord_length / wavelength))
    
    # 2. Generate the linear parameter 't' along the chord
    # Use many points for smooth plot rendering
    n_points = int( np.ceil( chord_length * 200 ))
    t = np.linspace(0, 1, n_points)
    one_side_straight_points = int( n_points * (1 - (wavelength * num_coils) / chord_length ) / 2 )
    
    # Base X and Y points along the straight chord
    x_base = x1 + t * (x2 - x1)
    y_base = y1 + t * (y2 - y1)
    
    # 3. Calculate the normal (perpendicular) vector to the chord
    dx, dy = x2 - x1, y2 - y1
    normal_x, normal_y = -dy, dx # Perpendicular vector
    
    # Normalize the normal vector
    magnitude = np.sqrt(normal_x**2 + normal_y**2)
    normal_x /= magnitude
    normal_y /= magnitude

    # 4. Create the Zigzag/Triangle wave offset
    # The 'width=0.5' argument ensures a perfect triangle wave (50% up-slope, 50% down-slope)
    # The 't' parameter needs to be scaled by the number of coils
    coil_wave_base = sawtooth(2 * np.pi * num_coils_maximum * t, width=0.5)
    coil_wave_base[:one_side_straight_points] = 0
    coil_wave_base[-one_side_straight_points:] = 0

    # 5. Integrate the straight ends (___)
    transition_factor = np.ones_like(t)
    
    final_coil_offset = coil_amplitude * coil_wave_base #* transition_factor
    
    # 6. Apply the final coil offset to the base line
    x_spring = x_base + normal_x * final_coil_offset
    y_spring = y_base + normal_y * final_coil_offset
    
    # 7. Convert the final Cartesian spring back to Polar (r, theta) for plotting
    r_spring = np.sqrt(x_spring**2 + y_spring**2)
    theta_spring = np.arctan2(y_spring, x_spring)

    # 8. Plot the spring
    ax.plot(theta_spring, r_spring, color=color, linewidth=1.5, zorder=5)

    return
