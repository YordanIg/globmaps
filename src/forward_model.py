"""
The entire forward-modelling formalism brought together.
"""
import numpy as np
from numba import jit

from src.indices_2BE.generate_2BE_indices import T_CMB
import src.coordinates as CO
import src.spherical_harmonics as SH
import src.beam_functions as BF
from src.sky_models import cm21_globalT
from src.blockmat import BlockMatrix, BlockVector

RS = SH.RealSphericalHarmonics()

def calc_averaging_matrix(Ntau, Nt):
    """
    Crude gain matrix to average up Nt neighbouring time bins into
    courser Ntau time bins. i.e. Ntau < Nt

    Exploits this trick for dividing each row by its own normalisation
    https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    """
    summation_matrix = np.zeros([Ntau, Nt])
    for i in range(Nt):
        value = int(i * Ntau / Nt)
        summation_matrix[value][i] = 1

    #normalise each row to get averaging matrix
    averaging_matrix = summation_matrix / np.sum(summation_matrix, axis=1)[:, None]

    return averaging_matrix

################################################################################
# Single-frequency observation matrices.
################################################################################

def calc_observation_matrix_zenith_driftscan(nside, lmax, Ntau=None, lat=-26, lon=0, 
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a single drifscan antenna pointing at zenith with a cos^2
    beamfunction. If Ntau = len(times), G is just the identity matrix.

    If return_mat is True, function returns tuple of G,P,Y,B too.
    """
    if Ntau is None:  # If Ntau isn't passed, don't bin.
        Ntau = len(times)
    coords = CO.obs_zenith_drift_scan(lat, lon, times)
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=len(times))
    mat_P = CO.calc_pointing_matrix(coords, nside=nside, pixels=False)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_P @ mat_Y @ mat_B, (mat_G, mat_P, mat_Y, mat_B)
    return mat_G @ mat_P @ mat_Y @ mat_B

def calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=None, lats=[-26],
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for multiple single drifscan antennas pointing at zenith with a cos^2
    beamfunction. If Ntau = len(times)*len(lats), G is just the identity matrix.

    If return_mat is True, function returns tuple of G,P,Y,B too.
    """
    if Ntau is None:  # If Ntau isn't passed, don't bin.
        Ntau = len(times)*len(lats)
    coords = [CO.obs_zenith_drift_scan(lat, lon=0, times=times) for lat in lats]
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=len(times)*len(lats))
    mat_P = CO.calc_pointing_matrix(*coords, nside=nside, pixels=False)
    try:
        mat_Y = np.load(f"saves/ylm_mat_nside{nside}_lmax{lmax}.npy")
    except:
        mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_P @ mat_Y @ mat_B, (mat_G, mat_P, mat_Y, mat_B)
    return mat_G @ mat_P @ mat_Y @ mat_B

def calc_observation_matrix_all_pix(nside, lmax, Ntau, beam_use=BF.beam_cos, 
                                    return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a hypothetical antenna experiment that can point at every pixel once.
    If Ntau = len(times), G is just the identity matrix.

    If spherical_harmonic_mat is True, function returns it too.
    """
    #pointing matrix is just the identity matrix, so not included
    npix = 12*nside**2
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=npix)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_Y @ mat_B, (mat_G, mat_Y, mat_B)
    return mat_G @ mat_Y @ mat_B

################################################################################
# Multi-frequency observation matrices.
################################################################################

def calc_observation_matrix_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=None, 
                            lat=-26, lon=0, 
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Do the same thing as calc_observation_matrix_zenith_driftscan but return 
    multifrequency block matrices.
    """
    mats = calc_observation_matrix_zenith_driftscan(nside, lmax, Ntau=Ntau, 
                            lat=lat, lon=lon, 
                            times=times, 
                            beam_use=beam_use, return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_P_bl = BlockMatrix(mat=mat_P, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_P_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mats, nblock=len(nuarr))

def calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=None, lats=[-26],
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Do the same thing as calc_observation_matrix_multi_zenith_driftscan but 
    return multifrequency block matrices.
    """
    mats = calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats,
                            times=times, 
                            beam_use=beam_use, return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_P_bl = BlockMatrix(mat=mat_P, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_P_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mats, nblock=len(nuarr))

def calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau=None, lats=[-26],
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos_FWHM, chromaticity=BF.fwhm_func_tauscher, return_mat=False):
    """
    Do the same thing as calc_observation_matrix_multi_zenith_driftscan_multifreq 
    but model beam chromaticity.
    """
    mats_A = []
    mats_G = []
    mats_P = []
    mats_Y = []
    mats_B = []
    for nu in nuarr:
        def beam(theta):
            return beam_use(theta, chromaticity(nu=nu))
        mat = calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats,
                            times=times, 
                            beam_use=beam, return_mat=return_mat)
        if return_mat:
            mat_A, (mat_G, mat_P, mat_Y, mat_B) = mat
            mats_A.append(mat_A)
            mats_G.append(mat_G)
            mats_P.append(mat_P)
            mats_Y.append(mat_Y)
            mats_B.append(mat_B)
        else:
            mats_A.append(mat)

    if return_mat:
        mat_A_bl = BlockMatrix(mat=np.array(mats_A))
        mat_G_bl = BlockMatrix(mat=np.array(mats_G))
        mat_P_bl = BlockMatrix(mat=np.array(mats_P))
        mat_Y_bl = BlockMatrix(mat=np.array(mats_Y))
        mat_B_bl = BlockMatrix(mat=np.array(mats_B))
        return mat_A_bl, (mat_G_bl, mat_P_bl, mat_Y_bl, mat_B_bl)
    
    return BlockMatrix(mat=np.array(mats_A))

def calc_observation_matrix_all_pix_multifreq(nuarr, nside, lmax, Ntau, 
                                              beam_use=BF.beam_cos, 
                                              return_mat=False):
    """
    Do the same thing as calc_observation_matrix_all_pix but return 
    multifrequency block matrices.
    """
    mats = calc_observation_matrix_all_pix(nside, lmax, Ntau, 
                                           beam_use=beam_use, 
                                           return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mat_A, nblock=len(nuarr))


################################################################################
# Alm polynomial foreground modelling.
################################################################################

def generate_alm_pl_forward_model(nuarr, observation_mat, Npoly=2, lmax=32):
    """
    Return a function that forward-models (without noise) the 
    Alm polynomial model. Note that this is a foreground only model.
    
    Returns
    -------
    model : function
        model is a function of a list of power law indices and runnings for
        each of the modelled sky alm up to and including lmax. These are ordered
        like (c_{00,0}, c_{00,1}, ..., c_{1-1,0}, c_{1-1,1}, ...)
        for c_{lm,n}, where n is the polynomial index.
    """
    if observation_mat.block_shape[1] != RS.get_size(lmax=lmax):
        raise ValueError(f"observation matrix size should correspond to lmax={lmax}")
    Nlmax = RS.get_size(lmax=lmax)
    Nnuarr = len(nuarr)
    
    def model(theta):
        # Compute the alm vector.
        theta_blocks = np.reshape(theta, (Nlmax, Npoly))
        alm_blocks = []
        for block in theta_blocks:
            A, alpha = block[:2]
            zetas    = block[2:]
            exponent = [zetas[i]*np.log(nuarr/60)**(i+2) for i in range(len(zetas))]
            alm_term = (A*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
            alm_blocks.append(alm_term)
        alm_blocks = np.array(alm_blocks)
        alm_blocks[0] += np.sqrt(4*np.pi)*T_CMB

        final_alm_vec = []
        for n in range(Nnuarr):
            final_alm_vec.append(alm_blocks[:,n])
        final_alm_vec = np.array(final_alm_vec)

        final_alm_vec = final_alm_vec.flatten()

        # Multiply this by the observation matrix.
        dmod = observation_mat @ final_alm_vec
        return dmod.vector

    return model

def genopt_alm_pl_forward_model(nuarr, observation_mat, Npoly=2, lmax=32):
    """
    Return a function that forward-models (without noise) the 
    Alm polynomial model. Note that this is a foreground only model.

    Same as generate_alm_pl_forward_model but uses jit for speedup.
    
    Returns
    -------
    model : function
        model is a function of a list of power law indices and runnings for
        each of the modelled sky alm up to and including lmax. These are ordered
        like (c_{00,0}, c_{00,1}, ..., c_{1-1,0}, c_{1-1,1}, ...)
        for c_{lm,n}, where n is the polynomial index.
    """
    if observation_mat.block_shape[1] != RS.get_size(lmax=lmax):
        raise ValueError(f"observation matrix size should correspond to lmax={lmax}")
    observation_mat = observation_mat.matrix
    Nlmax = RS.get_size(lmax=lmax)
    Nnuarr = len(nuarr)
    
    @jit
    def model(theta):
        # Compute the alm vector.
        theta_blocks = np.reshape(theta, (Nlmax, Npoly))
        alm_blocks = np.zeros((Nlmax, Nnuarr))
        for ii, block in enumerate(theta_blocks):
            A, alpha = block[:2]
            zetas    = np.zeros(len(block)-2)
            zetas    = block[2:]

            exponent = np.zeros((len(zetas), Nnuarr))
            for i in range(len(zetas)):
                exponent[i,:] = zetas[i]*np.log(nuarr/60)**(i+2)
            
            alm_term = (A*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
            alm_blocks[ii,:] = alm_term
        
        alm_blocks[0] += np.sqrt(4*np.pi)*T_CMB

        final_alm_vec = alm_blocks.T

        # Multiply this by the observation matrix.
        dmod = observation_mat @ final_alm_vec.flatten()
        return dmod

    return model

def genopt_alm_plfid_forward_model(nuarr, observation_mat, Npoly=2, lmod=1, lmax=32):
    """
    Return a function that forward-models (without noise) the 
    Alm polynomial model, correcting with the fiducial alm values above lmod. 
    Note that this is a foreground only model.
    
    Returns
    -------
    model : function
        model is a function of a list of power law indices and runnings for
        each of the modelled sky alm up to and including lmax. These are ordered
        like (c_{00,0}, c_{00,1}, ..., c_{1-1,0}, c_{1-1,1}, ...)
        for c_{lm,n}, where n is the polynomial index.
    """
    if observation_mat.block_shape[1] != RS.get_size(lmax=lmod):
        raise ValueError(f"observation matrix size should correspond to lmod={lmod}")
    observation_mat = observation_mat.matrix
    Nlmod = RS.get_size(lmax=lmod)
    Nnuarr = len(nuarr)
    
    @jit
    def model(theta):
        # Compute the alm vector.
        theta_blocks = np.reshape(theta, (Nlmod, Npoly))
        alm_blocks = np.zeros((Nlmod, Nnuarr))
        for ii, block in enumerate(theta_blocks):
            A, alpha = block[:2]
            zetas    = np.zeros(len(block)-2)
            zetas    = block[2:]

            exponent = np.zeros((len(zetas), Nnuarr))
            for i in range(len(zetas)):
                exponent[i,:] = zetas[i]*np.log(nuarr/60)**(i+2)
            
            alm_term = (A*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
            alm_blocks[ii,:] = alm_term
        
        alm_blocks[0] += np.sqrt(4*np.pi)*T_CMB
        final_alm_vec = alm_blocks.T

        # Multiply this by the observation matrix.
        dmod = observation_mat @ final_alm_vec.flatten()
        return dmod

    return model

def genopt_alm_plfid_forward_model_with21cm(nuarr, observation_mat, Npoly=2, lmod=1, lmax=32):
    """
    Return a function that forward-models (without noise) the 
    Alm polynomial model, correcting with the fiducial alm values above lmod. 
    Note that this is a foreground and 21-cm monopole model.
    
    Returns
    -------
    model : function
        model is a function of a list of power law indices and runnings for
        each of the modelled sky alm up to and including lmax. These are ordered
        like (c_{00,0}, c_{00,1}, ..., c_{1-1,0}, c_{1-1,1}, ...)
        for c_{lm,n}, where n is the polynomial index.
    """
    if observation_mat.block_shape[1] != RS.get_size(lmax=lmod):
        raise ValueError(f"observation matrix size should correspond to lmod={lmod}")
    observation_mat = observation_mat.matrix
    Nlmod = RS.get_size(lmax=lmod)
    Nnuarr = len(nuarr)
    
    @jit
    def model(theta):
        # Compute the alm vector.
        theta_cm21 = theta[-3:]
        theta_blocks = np.reshape(theta[:-3], (Nlmod, Npoly))
        alm_blocks = np.zeros((Nlmod, Nnuarr))
        for ii, block in enumerate(theta_blocks):
            A, alpha = block[:2]
            zetas    = np.zeros(len(block)-2)
            zetas    = block[2:]

            exponent = np.zeros((len(zetas), Nnuarr))
            for i in range(len(zetas)):
                exponent[i,:] = zetas[i]*np.log(nuarr/60)**(i+2)
            
            alm_term = (A*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
            alm_blocks[ii,:] = alm_term
        
        alm_blocks[0] += np.sqrt(4*np.pi)*(T_CMB + cm21_globalT(nu=nuarr, A=theta_cm21[0], nu0=theta_cm21[1], dnu=theta_cm21[2]))

        final_alm_vec = alm_blocks.T

        # Multiply this by the observation matrix.
        dmod = observation_mat @ final_alm_vec.flatten()
        return dmod

    return model

################################################################################
# Binwise foreground modelling.
################################################################################
def generate_dummy_mat_A(nuarr, Ntau, lmod):
    """
    Generates a dummy blockmatrix A full of zeros to pass to the functions below
    if you don't need to actually compute mat_A.
    """
    Nlmod = RS.get_size(lmax=lmod)
    mat_A_blocks = np.zeros((len(nuarr), Ntau, Nlmod))
    return BlockMatrix(mat=mat_A_blocks)

def generate_binwise_forward_model(nuarr, observation_mat: BlockMatrix, Npoly=2):
    # Determine the number of bins and number of frequencies, and make sure they
    # are all consistent.
    assert observation_mat.nblock == len(nuarr)
    Nfreq = len(nuarr)
    Nbin  = observation_mat.block_shape[0]

    def model(theta: np.ndarray):
        theta = np.reshape(theta, (Nbin, Npoly))
        polyvec = []
        for binpars in theta:
            for nu in nuarr:
                to_sum = [par*np.log(nu/60)**i for i, par in enumerate(binpars)]
                polyvec.append(np.sum(to_sum))
        polyvec = np.array(polyvec)
        polyvec = np.exp(polyvec)
        return polyvec + T_CMB
    return model

def generate_binwise_cm21_forward_model(nuarr, observation_mat: BlockMatrix, Npoly=2):
    # Determine the number of bins and number of frequencies, and make sure they
    # are all consistent.
    assert observation_mat.nblock == len(nuarr)
    Nfreq = len(nuarr)
    Nbin  = observation_mat.block_shape[0]

    def model(theta: np.ndarray):
        theta_fg = theta[:-3]
        theta_A, theta_nu0, theta_dnu = theta[-3:]
        cm21_mon = cm21_globalT(nuarr, theta_A, theta_nu0, theta_dnu)
        theta_fg = np.reshape(theta_fg, (Nbin, Npoly))
        polyvec = []
        for binpars in theta_fg:
            for nu in nuarr:
                to_sum = [par*np.log(nu/60)**i for i, par in enumerate(binpars)]
                polyvec.append(np.sum(to_sum))
        polyvec = np.array(polyvec)
        polyvec = np.exp(polyvec)
        return polyvec + cm21_mon + T_CMB
    return model

def genopt_binwise_cm21_forward_model(nuarr, observation_mat: BlockMatrix, Npoly=2):
    """
    Generate a binwise cm21 forward model with Numba JIT optimization.
    """
    # Determine the number of bins and number of frequencies, and make sure they
    # are all consistent.
    assert observation_mat.nblock == len(nuarr)
    Nfreq = len(nuarr)
    Nbin  = observation_mat.block_shape[0]

    @jit
    def model(theta):
        theta_fg = theta[:-3]
        theta_A, theta_nu0, theta_dnu = theta[-3:]
        cm21_mon = cm21_globalT(nuarr, theta_A, theta_nu0, theta_dnu)
        theta_fg = np.reshape(theta_fg, (Nbin, Npoly))
        
        polyvec = np.zeros(Nfreq * Nbin)
        idx = 0
        for bin_idx in range(Nbin):
            for nu_idx in range(Nfreq):
                nu = nuarr[nu_idx]
                to_sum = 0.0
                for i in range(Npoly):
                    to_sum += theta_fg[bin_idx, i] * np.log(nu / 60) ** i
                polyvec[idx] = np.exp(to_sum)
                idx += 1
        
        return polyvec + cm21_mon + T_CMB

    return model
