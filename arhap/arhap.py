from cycler import cycler

import numpy as np

from astropy.time import Time
from astropy import units as u
from astropy import constants as c

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

## Mission constants ##

# Kepler
Kepler_lamda = c.Constant('Kepler_lamda', "Kepler bandpass central wavelength",
                        6400, 'angstrom', 0.0, system='cgs',
                        reference="http://stev.oapd.inaf.it/~lgirardi/cmd_2.7/photsys.html")

Kepler_fwhm = c.Constant('Kepler_fwhm', "Kepler bandpass fwhm",
                        4000, 'angstrom', 0.0, system='cgs',
                        reference="https://archive.stsci.edu/kepler/manuals/KSCI-19033-001.pdf")

Kepler_zeropt = c.Constant('Kepler_zeropt', "Kepler flux at AB magnitude 12",
                           2.1*10**5, 'electron s-1', 0.0,
                           reference="https://archive.stsci.edu/kepler/manuals/KSCI-19033-002.pdf")

Kepler_lc_exptime = c.Constant('Kepler_lc_exptime', "Kepler long cadence exposure time",
                           1625.3467838829, 'second', 0.0,
                           reference="Conversation with Susan Mullally")

# GALEX
GALEX_fwhm = c.Constant('GALEX_fwhm', "GALEX bandpass fwhm",
                        795.65, 'angstrom', 0.0, system='cgs',
                        reference="http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.NUV")

GALEX_lamda = c.Constant('GALEX_lamda', "GALEX bandpass effective wavelength",
                         2304.74, 'angstrom', 0.0, system='cgs',
                         reference="http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.NUV")

## Time related ##

def galex_to_time(gtime):
    """
    Given a GALEX time (float) return and astropy time object.

    "GALEX Time" = "UNIX Time" - 315964800
    """
    return Time(gtime + 315964800, format='unix')

def time_to_galex(atime):
    """
    Given an astropy time object return the associated GALEX time (float)

    "GALEX Time" = "UNIX Time" - 315964800
    """

    return atime.unix - 315964800


## Magnitude/flux/luminosity related ##

def abmag_to_flux(mag, lambda_cent=None, fwhm=None):
    """
    Transform AB magnitude to flux in cgs units.

    If just a magnitude is given the returned flux is per unit frequency
    and has units erg s^−1 cm^−2 Hz^−1.

    If the central wavelength for a bandpass is given the returned flux
    will be given per unit wavelength with units erg s^−1 cm^−2 Angstrom^−1.

    If a FWHM of the bandpass is also given the flux will be calculated for
    the bandpass as a whole
    """
    
    # standard equation from Oke & Gunn (1883)
    f_nu = 10.0 ** ( (mag + 48.6) / (-2.5) ) * (u.Unit("erg s-1 cm-2 Hz-1"))
    if lambda_cent is None:
        return f_nu

    f_lambda = (f_nu * c.c / (lambda_cent*lambda_cent)).to("erg s-1 cm-2 angstrom-1")
    if fwhm is None:
        return f_lambda
    
    return (f_lambda * fwhm).to("erg s-1 cm-2")

def flux_to_abmag(flux, lambda_cent=None, fwhm=None):
    """
    Reverse of the above function. Currently does no checking so units need to be correct.
    """

    if not isinstance(flux, u.quantity.Quantity):
        if fwhm is not None:
            flux = flux * u.Unit("erg s-1 cm-2")
        elif lambda_cent is not None:
            flux = flux * u.Unit("erg s-1 cm-2 angstrom-1")
        else:
            flux = flux * u.Unit("erg s-1 cm-2 Hz-1")
            
    if fwhm is not None:
        f_lambda = flux/fwhm
    else:
        f_lambda = flux

    if lambda_cent is not None:
        f_nu = (f_lambda * (lambda_cent*lambda_cent) / c.c).to("erg s-1 cm-2 Hz-1")
    else:
        f_nu = flux.to("erg s-1 cm-2 Hz-1")

    # standard equation from Oke & Gunn (1883)
    mag = -2.5 * np.log10(f_nu.value) - 48.6

    return mag

    
def kepler_count_to_mag(flux):
     """
     Takes Kepler flux in electrons/sec (basically counts) and turns it into
     AB magnitude.

     This uses the benchmark photoelectron current at the Kepler focal plane for a 12th magnitude star: 
     
     flux_kep (mag_kep=12) = 2.1(10^5) e-/s
     
     from the Kepler Instrument Handbook (https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/kepler/_documents/KSCI-19033-002-instrument-hb.pdf).
     """

     if not isinstance(flux, u.quantity.Quantity):
         flux = flux * u.Unit("electron s-1")
     
     f_12 = 2.1 * 10**5 * u.Unit("electron s-1")

     return -2.5 * np.log10((flux/f_12).value) + 12  

 
def kepler_mag_to_count(mag):
     """
     Takes AB magnitude  and turns it into Kepler flux in electrons/sec (basically counts).

     This uses the benchmark photoelectron current at the Kepler focal plane for a 12th magnitude star: 
     
     flux_kep (mag_kep=12) = 2.1(10^5) e-/s
     
     from the Kepler Instrument Handbook (https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/kepler/_documents/KSCI-19033-002-instrument-hb.pdf).
     """

     f_12 = 2.1 * 10**5 * u.Unit("electron s-1")

     return f_12 * 10**(-0.4 * (mag - 12))

 
## Array related ##    

def find_nearest(arr, val):
    """
    Returns the index in arr whereto the value in arr that is closest to val. 
    """
    
    diff_arr = np.abs(arr - val)
    return np.where(diff_arr == diff_arr.min())[0][0]

def combine_intervals(data_table, start_col, end_col, max_dist):
    """
    Take a data table and two columns that represent the beginning/end of intervals,
    and the maximum distance between intervals to consider them continuous, and calculate
    the interval set.  data_table should be sorted on the start column.
    """

    interval_table = data_table[start_col, end_col]
    
    while True:
        neg_difs = (interval_table[start_col][1:] - (interval_table[end_col] + max_dist)[:-1]) < 0
        noneq_ints = (interval_table[start_col][1:] != interval_table[start_col][:-1]) | \
                     (interval_table[end_col][1:] != interval_table[end_col][:-1])
        overlap_inds = np.where(neg_difs & noneq_ints)[0]
        if len(overlap_inds) == 0:
            break
        for i in overlap_inds:
            interval_table[start_col][i+1] = interval_table[start_col][i]
            interval_table[end_col][i] = interval_table[end_col][i+1]
        
    return interval_table


## Plotting related ##

def align_yaxis(axes):
    """
    Align on zero any number of axes.

    PGlivi/Tim's solution from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    """
    
    y_lims = np.array([ax.get_ylim() for ax in axes])

    # force 0 to appear on all axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize all axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lims = y_new_lims_normalized * y_mags
    for i, ax in enumerate(axes):
        ax.set_ylim(new_lims[i])    



def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5):
    """
    Add error boxes to axis ax.

    From https://matplotlib.org/stable/gallery/statistics/errorbars_and_boxes.html (lightly modified).
    """

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)


def deduped_legend(ax, **kwargs):
    """
    From https://stackoverflow.com/a/56253636, with added kwarg pass-through.
    """
    
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), **kwargs)


def set_cycle_by_cmap(ax, cmap, length=50, cmap_lim=[0.1,0.9]):
    """
    Set color cycle of matplotlib axes object based on colormap.
    Adapted from https://tonysyu.github.io/mpltools/api/mpltools.color.html#cycle-cmap.
    
    Parameters
    ----------
    cmap : str
        Name of a matplotlib colormap (see matplotlib.pyplot.cm). 
    cmap_lim: arr
        Limit colormap to this range (0 <= start < stop <= 1). You should limit the
        range of colormaps with light values (assuming a white background).
    length : int
        The number of colors in the cycle. When `length` is large (> ~10), it
        is difficult to distinguish between successive lines because successive
        colors are very similar.
    ax : matplotlib axes
        Axes to apply the color cycle too
    """
    cmap = getattr(plt.cm, cmap)
    idx = np.linspace(cmap_lim[0], cmap_lim[1], num=length)
    color_cycle = cmap(idx)
    ax.set_prop_cycle(cycler(color=color_cycle))
