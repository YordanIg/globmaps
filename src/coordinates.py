"""
Code up conversion between spherical coordinate systems to allow
calculation of drift scan sky tracks

https://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf used as a reference
for the conversions.
"""
from numpy import cos, sin, arcsin, arccos, arctan2, pi
import numpy as np
import healpy as hp

deg2rad = pi / 180.0
rad2deg = 180.0 / pi
twopi = 2.0 * pi

def obs_zenith_drift_scan(lat, lon, times=np.linspace(0, 24, 24, endpoint=False)):
    """
    Return the galactic coordinate arrays (l, b) [deg] for
    an zenith-pointing antenna at (lat, lon) [deg] and 
    an array of hour-angle times.
    """
    rubbish = 1.0 #A unimportant for zenith observation
    ha = times * (360.0 / 24.0)   #local sidereal time
    alpha, delta = alt2eqa(rubbish, 0.0, lat, lon+ha)
    l, b = eqa2gal(alpha, delta)
    return l, b

def calc_pointing_matrix(*pointing_coordinates, nside=32, pixels=False):
    """
    Create pointing matrix corresponding to any number of
    sets of pointing coordinates.

    Parameters
    ----------
    pointing_coordinates : (2 x Npointing) - dimensional array-like
        Coordinate pointing vectors ((l1, l2, ...), (b1, b2, ...)) 
        for an antenna. Can pass one of these for each antenna used.
    nside : int
        The nside corresponding to the number of pixels of the 
        output matrix.
    pixels : bool
        If True, also returns the indexes of all pixels set to 1.

    Returns
    -------
    P : (sum(N_t for each antenna)) x (Npix) - dimension matrix
        The pointing matrix.
    """
    ls, bs = np.concatenate(pointing_coordinates, axis=1)
    thetas, phis = latlon2thetaphi(ls, bs)
    idxs = hp.pixelfunc.ang2pix(nside, phis*deg2rad, thetas*deg2rad)
    
    pointing_matrix = np.zeros((len(ls), hp.nside2npix(nside)))
    for row, idx in enumerate(idxs):
        pointing_matrix[row][idx] += 1
    
    if pixels:
        return pointing_matrix, idxs
    return pointing_matrix

def latlon2thetaphi(l, b):
    """
    Convert Galactic latitude and longitude l, b (deg) into spherical
    coordinates theta, phi (deg).
    """

    theta, phi = l, 90 - b  #healpy measures theta differently to latitude
    phi = (phi + 180) % 180
    return theta, phi

def eqa2gal(alpha_deg, delta_deg):
    """
    Convert equatorial coordinates alpha, delta (deg) to
    galactic coordinates l, b (deg).
    """

    alpha = alpha_deg * deg2rad
    delta = delta_deg * deg2rad

    #equinox 2000 coordinates
    d_NGP = 27.12 * deg2rad
    alpha0 = 282.86 * deg2rad
    l0 = 32.93 * deg2rad

    dalpha = alpha - alpha0
    sb = sin(delta) * sin(d_NGP) - cos(delta) * cos(d_NGP) * sin(dalpha)
    b = arcsin(sb)
    cdl = cos(dalpha) * cos(delta) / cos(b)
    sdl =(sin(delta) * cos(d_NGP) + cos(delta) * sin(d_NGP) * sin(dalpha)) / cos(b)

    dl = arctan2(sdl, cdl)
    l = dl + l0
    l = (l + twopi) % twopi

    return l * rad2deg, b * rad2deg

def gal2eqa(l_deg, b_deg):
    """
    Convert galactic coordinates l, b (deg) to
    equatorial coordinates alpha, delta (deg).
    """

    l = l_deg * deg2rad
    b = b_deg * deg2rad

    #equinox 2000 coordinates
    d_NGP = 27.12 * deg2rad
    alpha0 = 282.86 * deg2rad
    l0 = 32.93 * deg2rad

    dl = l - l0
    sd = sin(b) * sin(d_NGP) + cos(b) * cos(d_NGP) * sin(dl)
    delta = arcsin(sd)
    cda = cos(dl) * cos(b) / cos(delta)
    sda = (-sin(b) * cos(d_NGP) + cos(b) * sin(d_NGP) * sin(dl)) / cos(delta)

    da = arctan2(sda, cda)
    alpha = da + alpha0
    #alpha = alpha % (2.0 * pi)  #restrict alpha to [0,2pi]
    alpha = (alpha + twopi) % twopi

    return alpha * rad2deg, delta * rad2deg

def _test_gal_eqa():
    """
    Randomly test back and forth works for galactic and equilateral points
    """
    import numpy.random as npr
    tol = 1.0e-4

    value = True
    Ntest = 200
    for i in range(Ntest):
        a = npr.uniform(0.0, 360.0)
        d = npr.uniform(-90.0, 90.0)
        l, b = eqa2gal(a,d)
        a_out, d_out = gal2eqa(l, b)
        value1 = np.abs(a_out - a)<tol
        value2 = np.abs(d_out - d)<tol
        value = np.all([value, value1, value2])
    return value

def eqa2alt(alpha_deg, delta_deg, phi0_deg, theta0_deg):
    """
    Convert equatorial coordinates alpha, delta (deg) to Altazimuth A, z (deg)
    for an observer at latitude phi0 and longitude theta0 (deg).
    """

    alpha = alpha_deg * deg2rad
    delta = delta_deg * deg2rad
    phi0 = phi0_deg * deg2rad
    theta0 = theta0_deg * deg2rad

    ha = theta0 - alpha
    cz = sin(delta) * sin(phi0) + cos(delta) * cos(phi0) * cos(ha)
    z = arccos(cz)
    cA = (-sin(delta) * cos(phi0) + cos(delta) * sin(phi0) * cos(ha)) / sin(z)
    sA = sin(ha) * cos(delta) / sin(z)

    A = arctan2(sA, cA)

    return A * rad2deg, z * rad2deg

def alt2eqa(A_deg, z_deg, phi0_deg, theta0_deg):
    """
    Convert Altazimuth A, z (deg) to equatorial coordinates alpha, delta (deg).
    for an observer at latitude phi0 and longitude theta0 (deg).
    """
    A = A_deg * deg2rad
    z = z_deg * deg2rad
    phi0 = phi0_deg * deg2rad
    theta0 = theta0_deg * deg2rad

    sd = cos(z) * sin(phi0) - sin(z) * cos(phi0) * cos(A)
    delta = arcsin(sd)
    cha = (cos(z) * cos(phi0) + sin(z) * sin(phi0) * cos(A)) / cos(delta)
    sha = sin(A) * sin(z) / cos(delta)
    ha = arctan2(sha, cha)
    ha = (ha + twopi) % twopi #map from [-pi,pi] to [0,2pi]
    alpha = theta0 - ha
    alpha = (alpha + twopi) % twopi

    #alpha = alpha % (2.0 * pi)  #restrict alpha to [0,2pi]


    return alpha * rad2deg, delta * rad2deg

def _test_alt_eqa(phi0=-10.0, theta0 = 24.0):
    """
    Randomly test back and forth works for galactic and equilateral points
    """
    import numpy.random as npr
    tol = 1.0e-4

    value = True
    Ntest = 100
    for i in range(Ntest):
        a = npr.uniform(0.0, 360.0)
        d = npr.uniform(-90.0, 90.0)
        #print("in", a, d)
        A, z = eqa2alt(a,d, phi0, theta0)
        a_out, d_out = alt2eqa(A, z, phi0, theta0)
        #print("out", a_out, d_out)
        value1 = np.abs(a_out - a)<tol
        value2 = np.abs(d_out - d)<tol
        value = np.all([value, value1, value2])
    return value
