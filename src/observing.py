"""
Make actual observations!
"""
from functools import partial
import numpy as np

import src.beam_functions as BF
import src.forward_model as FM
import src.sky_models as SM
from src.spherical_harmonics import RealSphericalHarmonics
RS = RealSphericalHarmonics()

# Fiducial frequency array
Nfreq = 51
nuarr = np.linspace(50,100,Nfreq)

# Fiducial 21-cm parameters
cm21_params_old     = [-0.2, 80.0, 5.0]
cm21_params         = [-.13242, 68.57, 9.399]
cm21_dip_mag        = 1.2e-3
cm21_dip_centre     = np.radians([90-48.26, 263.99])
cm21_params_mondip  = [*cm21_params, cm21_dip_mag, *cm21_dip_centre]

# Default parameters of the observation and fiducial sky.
default_pars = {
    "times"  : np.linspace(0,6,3, endpoint=False),
    "unoise" : 1,                                         # Uniform noise level in Kelvin (only used if uniform_noise=True).
    "tint"   : 1,                                         # Total integration time in hours (only used if uniform_noise=False).
    "lmax"   : 32,
    "nside"  : 16,
    "lats"   : [-26]
}
ant_LUT = {
    1 : np.array([-26]),
    2 : np.array([-26, 26]),
    3 : np.array([-26, 0, 26]),
    4 : np.array([-26*2, -26, 26, 26*2]),
    5 : np.array([-26*2, -26, 0, 26, 26*2]),
    6 : np.array([-26*3, -26*2, -26, 26, 26*2, 26*3]),
    7 : np.array([-26*3, -26*2, -26, 0, 26, 26*2, 26*3])
}

def fiducial_obs(uniform_noise=False, unoise_K=None, tint=None, times=None, 
                 Ntau=None, lmax=None, nside=None, lats=None, delta=None, 
                 chrom=False, cm21_pars=None):
    """
    Forward model the fiducial degraded 2BE.

    Parameters
    ----------
    uniform_noise
        If uniform_noise is True, add gaussian uniform noise with level unoise_K 
        kelvin. If false, add radiometric noise with tint hours of total
        integration time.
    times
        Observation times in hours. Defaults to default_pars["times"].
    Ntau
        Number of observation bins. Defaults to len(times).
    lmax, nside
        Lmax and Nside of the fiducial sky. Defults to 
        default_pars["lmax/nside"].
    delta
        Gaussian width of uncertainty to add to the default 2BE indices - used
        to generate different "instances" of the foregrounds. Leave as None to 
        use the default 2BE.
    chrom
        Whether to use chromatic beams or not. If "None" or "False", no 
        chromaticity. If "True", uses the default value of c=2.4e-2 from 
        Tauscher et al 2020. Can also pass custon values for the c parameter.
    cm21_pars
        The Gaussian monopole parameters (A, nu0, dnu). If None, no monopole is
        included.
    """
    if unoise_K is None:
        unoise_K = default_pars["unoise"]
    if tint is None:
        tint = default_pars["tint"]
    if times is None:
        times = default_pars["times"]
    if Ntau is None:
        Ntau = len(times)
    if lmax is None:
        lmax = default_pars["lmax"]
    if nside is None:
        nside = default_pars["nside"]
    if lats is None:
        lats = default_pars["lats"]

    narrow_cosbeam = lambda x: BF.beam_cos(x, theta0=0.8)
    fid_alm, og_map = SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, use_mat_Y=True, nside=nside, original_map=True, delta=delta)
    if cm21_pars is not None:
        Nlmax = RS.get_size(lmax)
        cm21_a00 = np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *cm21_pars)
        fid_alm[::Nlmax] += cm21_a00
    
    if isinstance(chrom, bool):
        if not chrom:
            mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr=nuarr, nside=nside, lmax=lmax, Ntau=Ntau, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=True)
        elif chrom:
            mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=BF.fwhm_func_tauscher, return_mat=True)
    else:
        if chrom is None:
            mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr=nuarr, nside=nside, lmax=lmax, Ntau=Ntau, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=True)
        elif chrom is not None:
            chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
            mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc, return_mat=True)

    d = mat_A@(fid_alm)
    if uniform_noise:
        dnoisy, noise_covar = SM.add_noise_uniform(temps=d, err=unoise_K)
    elif not uniform_noise:
        dnoisy, noise_covar = SM.add_noise(temps=d, dnu=1, Ntau=Ntau, t_int=tint)

    params = {
        "uniform_noise" : uniform_noise,
        "unoise" : unoise_K,
        "tint"   : tint,
        "times"  : times,
        "Ntau"   : Ntau,
        "lmax"   : lmax,
        "nside"  : nside,
        "og" : og_map
    }
    return dnoisy, noise_covar, mat_A, mat_Y, params
