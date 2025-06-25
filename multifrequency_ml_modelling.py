"""
Using maximum-likelihood methods to reconstruct a_{00}(\nu), then fit a power
law and a 21-cm signal to it.
"""
# Import the ROOT_DIR from the config file.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ROOT_DIR

from functools import partial
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit
from emcee import EnsembleSampler
from numba import jit

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import src.map_making as MM
from src.blockmat import BlockMatrix, BlockVector
import src.inference as INF
import src.observing as OBS
from src.indices_2BE.generate_2BE_indices import T_CMB

alm2temp = 1/np.sqrt(4*np.pi)
RS = SH.RealSphericalHarmonics()

# Fit the foreground and 21-cm monopole.
def fg_polymod(nuarr, *theta_fg):
    Afg, alpha = theta_fg[:2]
    zetas      = theta_fg[2:]
    exponent = [zetas[i]*np.log(nuarr/60)**(i+2) for i in range(len(zetas))]
    fg_a00_terms = (Afg*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
    return fg_a00_terms + np.sqrt(4*np.pi)*T_CMB

@jit
def fg_polymod_opt(nuarr, *theta_fg):
    Afg = theta_fg[0]
    alpha = theta_fg[1]
    zetas = theta_fg[2:]
    exponent = np.zeros((len(zetas), len(nuarr)))
    for i in range(len(zetas)):
        for j in range(len(nuarr)):
            exponent[i, j] = zetas[i] * np.log(nuarr[j] / 60) ** (i + 2)
    fg_a00_terms = (Afg * 1e3) * (nuarr / 60) ** (-alpha) * np.exp(np.sum(exponent, axis=0))
    return fg_a00_terms + np.sqrt(4 * np.pi) * T_CMB

@jit
def cm21_mod(nuarr, *theta_21):
    A21, nu0, dnu = theta_21
    cm21_a00_terms = np.sqrt(4*np.pi) * A21 * np.exp(-.5*((nuarr-nu0)/dnu)**2)
    return cm21_a00_terms

@jit
def fg_cm21_polymod(nuarr, *theta):
    theta_fg = theta[:-3]
    theta_21 = theta[-3:]
    return fg_polymod_opt(nuarr, *theta_fg) + cm21_mod(nuarr, *theta_21)

################################################################################
def trivial_obs():
    # Model and observation params
    nside   = 16
    lmax    = 32
    lmod    = lmax
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    npix    = hp.nside2npix(nside)
    nuarr   = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm signal alm
    fg_alm   = SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix
    mat_A = FM.calc_observation_matrix_all_pix(nside, lmax, Ntau=npix, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
    if lmax != lmod:
        mat_A_mod = FM.calc_observation_matrix_all_pix(nside, lmod, Ntau=npix, beam_use=narrow_cosbeam)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
    else:
        mat_A_mod = mat_A

    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=npix, t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    _ = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod.block[-1], mat_N=noise_covar.block[-1], cond=True)
    mat_W   = MM.calc_ml_estimator_matrix(mat_A_mod, noise_covar)
    rec_alm = mat_W @ dnoisy

    # Extract the monopole component of the reconstructed alm.
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm.vector[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5, .001]
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, p0=fg_mon_p0+cm21_mon_p0)
    
    # Plot everything
    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:]), label='fit 21-cm monopole')
    plt.plot(nuarr, rec_a00-fg_a00, label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:])-cm21_mod(nuarr, *cm21_mon_p0), label='fit 21-cm monopole')
    plt.plot(nuarr, rec_a00-fg_a00-cm21_mod(nuarr, *cm21_mon_p0), label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0)-cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()


def nontrivial_obs_memopt_missing_modes(Npoly=3, lats=None, chrom=None, 
                                        basemap_err=5, err_type='idx', 
                                        mcmc=False, mcmc_pos=None, savetag="", 
                                        numerical_corr=False, steps=10000, 
                                        burn_in=3000, plotml=True, lmax=None, 
                                        lmod=None):
    """
    A memory-friendly version of nontrivial_obs which computes the reconstruction
    of each frequency seperately, then brings them all together.

    Works for chromatic and achromatic observations, using a narrowed cosine
    beam or a cosine beam with frequency varying width, depending on the 
    chromaticity case.

    For some reason when Ntau!=None the ML reconstruction doesn't work.
    """
    # Mapmaking pipeline parameters.
    nside   = 32
    if lmax is None:
        lmax = 32
    if lmod is None:
        lmod = 5
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)

    # Observation and binning params.
    Ntau  = None
    Nt    = 24
    times = np.linspace(0, 24, Nt, endpoint=False)
    nuarr = np.linspace(50,100,51)
    if lats is None:
        lats = np.array([-26*3, -26*2, -26, 0, 26, 26*2, 26*3])

    # Cosmological parameters.
    cm21_params = OBS.cm21_params
    
    # Foreground correction reference frequency.
    err_ref = 70

    # Generate foreground and 21-cm signal alm
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fg_alm   = SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, 
        use_mat_Y=True, delta=SM.basemap_err_to_delta(basemap_err, ref_freq=err_ref), 
        err_type=err_type, seed=100, meancorr=False)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the modelling and for the observations.
    if chrom is not None:
        if not isinstance(chrom, bool):
            chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
        else:
            chromfunc = BF.fwhm_func_tauscher
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau=Ntau, lats=lats, times=times, return_mat=False, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
        mat_A_mod = mat_A[:,:Nlmod]
    elif chrom is None:
        narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=False)
        mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
        mat_A_mod = mat_A[:,:Nlmod]
    
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=200, seed=456)#t_int=100, seed=456)#
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    # Calculate the missing-modes observation matrix.
    mat_A_unmod = BlockMatrix(mat_A.block[:,:,Nlmod:])

    # Generate a missing-modes correction numerically by generating instances of 
    # the 2be and finding the mean and covariance.
    if numerical_corr:
        fg_alm_list = []
        for i in range(100):
            if err_type=='idx':
                fg_alm_list.append(SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=SM.basemap_err_to_delta(basemap_err, ref_freq=err_ref), err_type=err_type, seed=123+i, meancorr=False))
            else:
                fg_alm_list.append(SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=basemap_err, err_type=err_type, seed=123+i))
        fg_alm_arr = np.array(fg_alm_list)
        fg_alm_arr = np.array(np.split(fg_alm_arr, len(nuarr), axis=1))
        fg_alm_unmod_arr  = fg_alm_arr[:,:,Nlmod:]
        # Step 3: Compute the data correction and covariance correction.
        data_corr  = []
        covar_corr = []
        for alm_block, mat_A_unmod_block in zip(fg_alm_unmod_arr, mat_A_unmod.block):
            alm_block_mean = np.mean(alm_block, axis=0)
            alm_block_cov  = np.cov(alm_block, rowvar=False)
            data_corr.append(mat_A_unmod_block @ alm_block_mean)
            covar_corr.append(mat_A_unmod_block @ alm_block_cov @ mat_A_unmod_block.T)
        data_corr = BlockVector(np.array(data_corr))
        covar_corr = BlockMatrix(np.array(covar_corr))
    
    # Generate a missing-modes correction analytically.
    elif not numerical_corr:
        alm_mean, alm_cov = SM.corr_2be(lmod, lmax, nside, nuarr, basemap_err, ref_freq=err_ref)
        data_corr = mat_A_unmod @ alm_mean
        covar_corr = mat_A_unmod @ alm_cov @ mat_A_unmod.T

    # Reconstruct the max likelihood estimate of the alm
    mat_W, cov = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod, mat_N=noise_covar+covar_corr, cov=True, cond=True)
    if isinstance(cov, BlockMatrix):
        alm_error = np.sqrt(cov.diag)
    else:
        alm_error = np.sqrt(np.diag(cov))
    print("Computing rec alm")
    if isinstance(mat_W, BlockMatrix):
        rec_alm = mat_W @ (dnoisy - data_corr)
    else:
        rec_alm = mat_W @ (dnoisy - data_corr).vector

    # Compute the chi-square and compare it to the length of the data vector.
    print("Computing chi-sq")
    chi_sq = ((dnoisy - data_corr) - mat_A_mod@rec_alm).T @ noise_covar.inv @ ((dnoisy - data_corr) - mat_A_mod@rec_alm)
    chi_sq = sum(chi_sq.diag)
    print("Chi-square:", chi_sq, "len(data):", dnoisy.vec_len,"+/-", np.sqrt(2*dnoisy.vec_len), "Nparams:", Nlmod*len(nuarr))
    
    # Extract the monopole component of the reconstructed alm.
    if isinstance(rec_alm, BlockVector):
        rec_a00 = np.array(rec_alm.vector[::Nlmod])
    else:
        rec_a00 = np.array(rec_alm[::Nlmod])
    a00_error = np.array(alm_error[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5]
    fg_mon_p0 += [.001]*(Npoly-2)
    cm21_mon_p0 = cm21_params
    bounds = [[1, 25], [1.5, 3.5]]
    bounds += [[-2, 2.1]]*(Npoly-2)
    bounds += [[-0.4, -0.02], [62, 88], [6, 14]]
    bounds = list(zip(*bounds))
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=fg_mon_p0+cm21_mon_p0, bounds=bounds)

    if plotml:
        _plot_results(nuarr, Nlmax, Nlmod, rec_alm.vector, alm_error, fid_alm, cm21_alm, res)

    if mcmc:
        def mod(theta):
            return fg_cm21_polymod(nuarr, *theta)
        
        # create a small ball around the MLE to initialize each walker
        nwalkers, fg_dim = 64, Npoly+3
        ndim = fg_dim
        if mcmc_pos is not None:
            pos = mcmc_pos*(1 + 1e-4*np.random.randn(nwalkers, ndim))
        else:
            pos = res[0]*(1 + 1e-4*np.random.randn(nwalkers, ndim))
        priors = [[1, 25], [1.5, 3.5]]
        priors += [[-2, 2.1]]*(Npoly-2)
        priors += [[-0.5, -0.01], [60, 90], [5, 15]]
        priors = np.array(priors)
        # run emcee without priors
        sampler = EnsembleSampler(nwalkers, ndim, INF.log_posterior, 
                            args=(rec_a00, a00_error, mod, priors))
        _=sampler.run_mcmc(pos, nsteps=steps, progress=True, skip_initial_state_check=True)
        chain_mcmc = sampler.get_chain(flat=True, discard=burn_in)

        prestr = f"Nant<{len(lats)}>_Npoly<{Npoly}>_"
        if chrom is None:
            prestr += "achrom_"
        else:
            prestr += "chrom<{:.1e}>_".format(chrom)
        if basemap_err is not None:
            prestr += err_type+"<{}>_".format(basemap_err)
    
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"mcmcChain.npy", chain_mcmc)
        
        # Calculate the BIC for MCMC.
        c = ChainConsumer()
        c.add_chain(chain_mcmc, statistics='max')
        analysis_dict = c.analysis.get_summary(squeeze=True)
        theta_max = np.array([val[1] for val in analysis_dict.values()])
        loglike = INF.log_likelihood(theta_max, y=rec_a00, yerr=a00_error, model=mod)
        bic = len(theta_max)*np.log(len(rec_a00)) - 2*loglike
        print("bic is ", bic)
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"bic.npy", bic)

        # Calculate the total model residuals and save them for plotting.
        fid_a00 = fid_alm[::Nlmax]
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"modres.npy", ((dnoisy-data_corr) - mat_A_mod@rec_alm).vector)
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"data.npy", dnoisy.vector)
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"dataerr.npy", np.sqrt(noise_covar.diag+covar_corr.diag))
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"fid_a00.npy", fid_a00)
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"rec_a00.npy", rec_a00)
        np.save(str(ROOT_DIR)+"/saves/MLmod/"+prestr+savetag+"rec_a00_err.npy", a00_error)
    del mat_A
    del mat_A_mod
    del mat_A_unmod


def _plot_results(nuarr, Nlmax, Nlmod, rec_alm, alm_error, fid_alm, cm21_alm, final_fitres):
    fg_alm = fid_alm-cm21_alm

    # Extract the monopole component of the reconstructed alm.
    fid_a00  = np.array(fid_alm[::Nlmax])
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm[::Nlmod])
    a00_error = np.array(alm_error[::Nlmod])

    # Plot the reconstructed a00 mode minus the fiducial a00 mode.
    plt.plot(nuarr, rec_a00-fid_a00, label='$a_{00}$ reconstructed - $a_{00}$ fid fg')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    # Plot the reconstructed a00 mode minus the best-fitting power law with no
    # running.
    res = curve_fit(fg_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=[15,2.5])
    plt.plot(nuarr, rec_a00-fg_polymod(nuarr, *res[0]), label='$a_{00}$ reconstructed - power law')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    # Provide a corner plot for the 21-cm inference.
    # Draw samples from the likelihood.
    chain = np.random.multivariate_normal(mean=final_fitres[0][-3:], cov=final_fitres[1][-3:,-3:], size=100000)
    c = ChainConsumer()
    c.add_chain(chain, parameters=['A', 'nu0', 'dnu'])
    f = c.plotter.plot()
    plt.show()

    # Evaluate the model at 1000 points drawn from the chain to get 1sigma 
    # inference bounds in data space.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    chain_samples = np.random.multivariate_normal(mean=final_fitres[0][-3:], cov=final_fitres[1][-3:,-3:], size=1000)
    cm21_a00_sample_list = [cm21_a00_mod(nuarr, theta) for theta in chain_samples]
    cm21_a00_sample_mean = np.mean(cm21_a00_sample_list, axis=0)
    cm21_a00_sample_std = np.std(cm21_a00_sample_list, axis=0)

    # Plot the model evaluated 1 sigma regions and the fiducial monopole.
    cm21_a00 = np.array(cm21_alm[::Nlmax])
    plt.plot(nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std, 
        cm21_a00_sample_mean+cm21_a00_sample_std,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()

    # Do the same thing but take the residuals.
    plt.plot(nuarr, cm21_a00-cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Inferred 21-cm a00 residuals [K]")
    plt.legend()
    plt.show()


#################################################
# Running the function with example parameters. #
#################################################

def gen_ml_chrom(Nant=4, Npoly=6, chrom=None, basemap_err=None):
    nontrivial_obs_memopt_missing_modes(Npoly=Npoly, lats=OBS.ant_LUT[Nant], 
                                        chrom=chrom, basemap_err=basemap_err, 
                                        err_type='idx', mcmc=True, 
                                        mcmc_pos=None, steps=100000, 
                                        burn_in=60000, plotml=False)

# Achromatic, no basemap error.
def run_set_gen_ml_chrom0_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=0)

# c=0 chromaticity, no basemap error.
def run_set_gen_ml_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=0)

# 1.6e-2 chromaticity, no basemap error.
def run_set_gen_ml_chromsmall_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=0)

# 3.4e-2 chromaticity, no basemap error.
def run_set_gen_ml_chrom_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=0)

# Achromatic, 5% basemap error.
def run_set_gen_ml_chrom0_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=5)

# c=0 chromaticity, 5% basemap error.
def run_set_gen_ml_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=5)

# 1.6e-2 chromaticity, 5% basemap error.
def run_set_gen_ml_chromsmall_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=5)

# 3.4e-2 chromaticity, 5% basemap error.
def run_set_gen_ml_chrom_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=5)

# Achromatic, 10% basemap error.
def run_set_gen_ml_chrom0_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=10)

# 1.6e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chromsmall_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=10)

# 3.4e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chrom_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=10)

# 3.4e-2 chromaticity, 20% basemap error.
def run_set_gen_ml_chrom_bm20(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=20)

# 5.2e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chromlarge_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=5.2e-2, basemap_err=10)

######################################
# Producing plots of the above runs. #
######################################

def plot_ml_chrom(Nant=7, Npoly=7, chromstr=None, basemap_err=None, savetag=None):
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_idx<{basemap_err}>"
    mcmcChain = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_mcmcChain.npy')
    residuals = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_modres.npy')
    data      = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_data.npy')
    dataerr   = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_dataerr.npy')
    fid_a00   = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr+"_fid_a00.npy")
    rec_a00   = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr+"_rec_a00.npy")
    a00_error = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr+"_rec_a00_err.npy")

    try:
        bic=np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant))
    data  = np.reshape(data, (Nfreq, Nant, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant, Ntau))
    
    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(mcmcChain[:,-3:], parameters=[r'$A_{21}$', r'$nu_{21}$', r'$\Delta$'])
    f = c.plotter.plot(truth=[*OBS.cm21_params])
    if savetag is not None:
        f.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+"_corner.pdf")
        f.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+"_corner.png")
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(4,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel(r"21-cm Temperature [K]")
    ax[0].legend()

    ax[1].axhline(y=0, linestyle=':', color='k')
    ax[1].errorbar(OBS.nuarr, (rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k')
    ax[1].axhline(0, linestyle=':', color='k')
    ax[1].set_ylabel(r"$\hat{T}_\mathrm{mon}-\mathcal{M}$ [K]")
    fig.tight_layout()
    if savetag is not None:
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+".pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+".png")
    plt.show()

    plt.errorbar(OBS.nuarr, (fid_a00-rec_a00)*alm2temp, a00_error*alm2temp, fmt='.')
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"$T_\mathrm{mon} - \hat{T}_\mathrm{mon}$ [K]")
    if savetag is not None:
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+"_inferred_Tmon_res.pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/ml_"+runstr+savetag+"_inferred_Tmon_res.png")
    plt.show()
    
    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc**2)
    print("monopole chi-sq", chi_sq)
    np.save(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_chi_sq.npy', chi_sq)


def plot_ml_chi_sq_bic(Nant=4, Npolys=[], chromstr='3.4e-02', basemap_err=None, savetag=None):
    runstrs = []
    for Npoly in Npolys:
        runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
        if chromstr is not None:
            runstr += f"_chrom<{chromstr}>"
        else:
            runstr += f"_achrom"
        if basemap_err is not None:
            runstr += f"_idx<{basemap_err}>"
        runstrs.append(runstr)
    chi_sqs = []
    bics    = []
    for runstr in runstrs:
        chi_sqs.append(np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_chi_sq.npy'))
        bics.append(np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_bic.npy'))

    fig, ax1 = plt.subplots()
    ax1.axhline(y=1, linestyle=':', color='k')
    ax1.semilogy(Npolys,chi_sqs, color='C0', linestyle='-', marker='o')
    ax1.set_xticks(ticks=Npolys, labels=Npolys)
    ax1.set_xticks(ticks=[], minor=True)
    ax1.set_ylabel(r"21-cm Monpole $\chi^2$")
    ax1.set_xlabel("$N_\mathrm{poly}$")
    ax1.set_xlim([Npolys[0], Npolys[-1]])
    ax2 = ax1.twinx()
    ax2.semilogy(Npolys,bics, color='C1', linestyle='-', marker='s')
    ax2.set_ylabel("Model BIC")
    custom_lines = [
        Line2D([0], [0], color='C0', linestyle='-', marker='o'),
        Line2D([0], [0], color='C1', linestyle='-', marker='s')
    ]
    # Add the custom legend to the plot
    plt.legend(custom_lines, [r'$\chi^2$', 'BIC'])
    fig.tight_layout()
    if savetag is not None:
        s = f"ml_chi_sq_bic_Nant<{Nant}>"
        if chromstr is not None:
            s += f"_chrom<{chromstr}>"
        else:
            s += "_achrom"
        if basemap_err is not None:
            s += f"_idx<{basemap_err}>"
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/"+s+savetag+".pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/"+s+savetag+".png")
    plt.show()

# Chromaticity: None (achromatic), Basemap error: 0
def plot_set_ml_chrom0_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=0, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=0, savetag=savetag)

# Chromaticity: 0 (flat), Basemap error: 0
def plot_set_ml_chromflat_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chromflat_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='0.0e+00', basemap_err=0, savetag=savetag)

# Chromaticity: 1.6e-2, Basemap error: 0
def plot_set_ml_chromsmall_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chromsmall_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=0, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 0
def plot_set_ml_chrom_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chrom_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=0, savetag=savetag)

# Chromaticity: None (achromatic), Basemap error: 5
def plot_set_ml_chrom0_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=5, savetag=savetag)

# Chromaticity: 0 (flat), Basemap error: 5
def plot_set_ml_chromflat_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chromflat_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='0.0e+00', basemap_err=5, savetag=savetag)

# Chromaticity: 1.6e-2, Basemap error: 5
def plot_set_ml_chromsmall_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chromsmall_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 5
def plot_set_ml_chrom_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=5, savetag=savetag)

# Chromaticity: None (achromatic), Basemap error: 10
def plot_set_ml_chrom0_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=10, savetag=savetag)

# Chromaticity: 1.6e-2, Basemap error: 10
def plot_set_ml_chromsmall_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chromsmall_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=10, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 10
def plot_set_ml_chrom_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=10, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 20
def plot_set_ml_chrom_bm20(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=20, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm20(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=20, savetag=savetag)

# Chromaticity: 5.2e-2, Basemap error: 10
def plot_set_ml_chromlarge_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='5.2e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chromlarge_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='5.2e-02', basemap_err=10, savetag=savetag)

