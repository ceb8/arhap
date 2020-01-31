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
