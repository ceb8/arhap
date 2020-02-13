import numpy as np

from astropy.time import Time


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
