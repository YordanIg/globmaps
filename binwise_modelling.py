"""
Doing binwise multifrequency modelling to see if any of the methods we're
developing actually perform better.
"""
# Import the ROOT_DIR from the config file.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ROOT_DIR

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit
from emcee import EnsembleSampler
from chainconsumer import ChainConsumer

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import src.inference as INF
import src.observing as OBS
from src.blockmat import BlockMatrix

RS = SH.RealSphericalHarmonics()
alm2temp = 1/np.sqrt(4*np.pi)


def fg_cm21_chrom_corr(Npoly=3, chrom=None, basemap_err=None, savetag=None, times=None, lats=None, mcmc_pos=None, steps=3000, burn_in=1000):
    """

    """
    # Model and observation params
    if lats is None:
        lats  = np.array([-26*3, -26*2, -26, 0, 26, 26*2, 26*3])
    if times is None:
        times = np.linspace(0, 24, 12, endpoint=False)

    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    Ntau    = 1
    nuarr   = np.linspace(50,100,51)
    cm21_params = OBS.cm21_params

    # Generate foreground and 21-cm alm
    fg_alm = SM.foreground_2be_alm_nsidelo(
        nu=nuarr, lmax=lmax, nside=nside, 
        use_mat_Y=True, 
        delta=SM.basemap_err_to_delta(basemap_err), err_type='idx'
    )
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the observations.
    if chrom is None:
        # Generate observation matrix for the achromatic case.
        narrow_cosbeam = lambda x: BF.beam_cos(x, 0.8)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=True)
        mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
        mat_G = BlockMatrix(mat=mat_G, mode='block', nblock=len(nuarr))
        mat_P = BlockMatrix(mat=mat_P, mode='block', nblock=len(nuarr))
        mat_Y = BlockMatrix(mat=mat_Y, mode='block', nblock=len(nuarr))
        mat_B = BlockMatrix(mat=mat_B, mode='block', nblock=len(nuarr))
    else:
        # Generate observation matrix for the chromatic case.
        chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau=Ntau, lats=lats, times=times, return_mat=True, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)

    # Perform fiducial observations without binning.
    d = mat_P @ mat_Y @ mat_B @ fid_alm

    # NOTE: using len(times) here because we keep the data unbinned to apply the 
    # chromaticity correction. Noise is recomputed later.
    dnoisy, noise_covar = SM.add_noise(d, nuarr[1]-nuarr[0], len(times), t_int=200) 
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    dnoisy_vector = dnoisy.vector
    
    # Perform an EDGES-style chromaticity correction.
    if chrom is not None:
        # Generate alm of the Haslam-shifted sky and observe them using our beam.
        has_alm = SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmax, 
            nside=nside, use_mat_Y=True, 
            const_idx=True, 
            delta=basemap_err, 
            err_type='bm', seed=124
        )
        chrom_corr_numerator = mat_P @ mat_Y @ mat_B @ has_alm
        # Construct an observation matrix of the hypothetical (non-chromatic) case.
        mat_B_ref = BlockMatrix(mat=mat_B.block[10], nblock=mat_B.nblock)
        chrom_corr_denom = mat_P @ mat_Y @ mat_B_ref @ has_alm
        chrom_corr = chrom_corr_numerator.vector/chrom_corr_denom.vector
        dnoisy_vector /= chrom_corr

    # Bin the noisy data and the noise.
    dnoisy = mat_G @ dnoisy_vector
    noise = np.std([np.sqrt(np.diag(noise_covar.block[n])) for n in range(noise_covar.nblock)], axis=1)

    # Set up the foreground model
    mod = FM.genopt_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    mod_prerun = FM.generate_binwise_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod_prerun(theta)

    # Try curve_fit, if it doesn't work just set res to the guess parameters.
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=noise)
    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))

    # create a small ball around the MLE to initialize each walker
    nwalkers, fg_dim = 64, Npoly+3
    ndim = fg_dim
    if mcmc_pos is not None:
        if len(mcmc_pos) < ndim:
            print("error - mcmc start pos has too few params. Adding some.")
            params_to_add = [0.01]*(ndim-len(mcmc_pos))
            mcmc_pos = np.append(mcmc_pos[:-3], params_to_add)
            mcmc_pos = np.append(mcmc_pos, cm21_params)
        elif len(mcmc_pos) > ndim:
            print("error - mcmc start pos has too many params. Cutting some.")
            new_pos = mcmc_pos[:Npoly]
            new_pos = np.append(new_pos, cm21_params)
            mcmc_pos = np.array(new_pos)
        p0 = mcmc_pos
    else:
        p0 = np.append(res[0], OBS.cm21_params)
    priors = [[1, 25], [-3.5, -1.5]]
    priors += [[-10, 10.1]]*(Npoly-2)
    priors += [[-2, -0.001], [60, 90], [5, 15]]
    priors = np.array(priors)
    
    p0 = INF.prior_checker(priors, p0)
    pos = p0*(1 + 1e-4*np.random.randn(nwalkers, ndim))
    err = noise
    sampler = EnsembleSampler(nwalkers, ndim, INF.log_posterior, 
                        args=(dnoisy.vector, err, mod, priors))
    _=sampler.run_mcmc(pos, nsteps=steps, progress=True, skip_initial_state_check=True)
    chain_mcmc = sampler.get_chain(flat=True, discard=burn_in)

    # Calculate the BIC for MCMC.
    bic = None
    c = ChainConsumer()
    c.add_chain(chain_mcmc, statistics='max')
    analysis_dict = c.analysis.get_summary(squeeze=True)
    theta_max = np.array([val[1] for val in analysis_dict.values()])
    loglike = INF.log_likelihood(theta_max, y=dnoisy.vector, yerr=err, model=mod)
    bic = len(theta_max)*np.log(len(dnoisy)) - 2*loglike
    print("bic is ", bic)

    if savetag is not None:
        prestr = f"Nant<{len(lats)}>_Npoly<{Npoly}>"
        if chrom is None:
            prestr += "_achrom"
        else:
            prestr += "_chrom<{:.1e}>".format(chrom)
        if basemap_err is not None:
            prestr += '_bm'+f"<{basemap_err}>"
    
        np.save(str(ROOT_DIR)+"/saves/Binwise/"+prestr+savetag+"_data.npy", dnoisy.vector)
        np.save(str(ROOT_DIR)+"/saves/Binwise/"+prestr+savetag+"_dataerr.npy", noise)
        np.save(str(ROOT_DIR)+"/saves/Binwise/"+prestr+savetag+"_mcmcChain.npy", chain_mcmc)
        if bic is not None:
            print("saving bic")
            np.save(str(ROOT_DIR)+"/saves/Binwise/"+prestr+savetag+"_bic.npy", bic)


#################################################
# Running the function with example parameters. #
#################################################

def gen_binwise_chrom(Nant=4, Npoly=4, chrom=None, basemap_err=None, savetag=None):
    fg_cm21_chrom_corr(Npoly=Npoly, chrom=chrom, savetag=savetag, 
                       lats=OBS.ant_LUT[Nant], mcmc_pos=None, 
                       basemap_err=basemap_err, steps=200000, burn_in=175000)

# Achromatic, no basemap error.
def run_set_gen_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=None, savetag=savetag)

# c=0 chromaticity, no basemap error.
def run_set_gen_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=None, savetag='')

# 1.6e-2 chromaticity, no basemap error.
def run_set_gen_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=None, savetag=savetag)

# 3.4e-2 chromaticity, no basemap error.
def run_set_gen_binwise_chrom_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=None, savetag=savetag)

# Achromatic, 5% basemap error.
def run_set_gen_binwise_chrom0_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=5, savetag=savetag)

# c=0 chromaticity, 5% basemap error.
def run_set_gen_binwise_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=5, savetag='')

# 1.6e-2 chromaticity, 5% basemap error.
def run_set_gen_binwise_chromsmall_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=5, savetag=savetag)

# 3.4e-2 chromaticity, 5% basemap error.
def run_set_gen_binwise_chrom_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=5, savetag=savetag)


######################################
# Producing plots of the above runs. #
######################################

def plot_binwise_chrom(Nant=7, Npoly=7, chromstr='3.4e-02', basemap_err=None, ml_plots=False, savetag=None):
    if ml_plots:
        print("Warning: ML posteriors are not being generated for new runs - this may fail/produce unexpected results.")
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_bm<{basemap_err}>"
    if savetag is not None:
        runstr += savetag
    print("loading from", runstr)
    mcmcChain = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_mcmcChain.npy')
    if ml_plots:
        mlChain   = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_mlChain.npy')
    data      = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_data.npy')
    dataerr   = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_dataerr.npy')

    try:
        bic=np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(mcmcChain[:,-3:], parameters=['A', 'nu0', 'dnu'])
    if ml_plots:
        c.add_chain(mlChain)
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)
    if ml_plots:        
        idx_mlChain = np.random.choice(a=list(range(len(mlChain))), size=10000)
        samples_mlChain = mlChain[idx_mlChain]
        a00list_ml   = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mlChain]
        a00mean_ml = np.mean(a00list_ml, axis=0)
        a00std_ml  = np.std(a00list_ml, axis=0)

        plt.plot(OBS.nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
        plt.fill_between(
            OBS.nuarr,
            a00mean_ml-a00std_ml, 
            a00mean_ml+a00std_ml,
            color='C1',
            alpha=0.8,
            edgecolor='none',
            label="inferred"
        )
        plt.fill_between(
            OBS.nuarr,
            a00mean_ml-2*a00std_ml, 
            a00mean_ml+2*a00std_ml,
            color='C1',
            alpha=0.4,
            edgecolor='none'
        )
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("21-cm a00 [K]")
        plt.legend()
        plt.show()

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
    ax[1].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("21-cm Temperature [K]")
    ax[0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[0].legend()

    mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
    mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npoly)
    ax[1].axhline(y=0, linestyle=':', color='k')
    ax[1].errorbar(OBS.nuarr, mod(np.mean(mcmcChain, axis=0))-data, dataerr, fmt='.', color='k')
    ax[1].set_ylabel(r"$T_\mathrm{res}$ [K]")
    fig.tight_layout()
    if savetag is not None:
        plt.savefig(str(ROOT_DIR)+"/fig/Binwise/bw_"+runstr+savetag+".pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/Binwise/bw_"+runstr+savetag+".png")

    plt.show()

    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc**2)
    print("monopole chi-sq", chi_sq)
    np.save(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_chi_sq.npy', chi_sq)


def plot_binwise_chi_sq_bic(Nant=7, Npolys=[], chromstr='3.4e-02', basemap_err=None, savetag=None):
    runstrs = []
    for Npoly in Npolys:
        runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
        if chromstr is not None:
            runstr += f"_chrom<{chromstr}>"
        else:
            runstr += f"_achrom"
        if basemap_err is not None:
            runstr += f"_bm<{basemap_err}>"
        runstrs.append(runstr)
    chi_sqs = []
    bics    = []
    for runstr in runstrs:
        print("loading from", runstr)
        chi_sqs.append(np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_chi_sq.npy'))
        bics.append(np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_bic.npy'))

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
        s = f"bw_chi_sq_bic_Nant<{Nant}>"
        if chromstr is not None:
            s += f"_chrom<{chromstr}>"
        else:
            s += "_achrom"
        if basemap_err is not None:
            s += f"_bm<{basemap_err}>"
        plt.savefig(str(ROOT_DIR)+"/fig/Binwise/"+s+savetag+".pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/Binwise/"+s+savetag+".png")
    plt.show()


# Chromaticity: None (achromatic), Basemap error: None
def plot_set_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=None, savetag=savetag)

# Chromaticity: None (achromatic), Basemap error: None
def plot_binwise_chi_sq_bic_chrom0_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=None, savetag=savetag)

# Chromaticity: 0 (flat), Basemap error: None
def plot_set_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=None)

# Chromaticity: 1.6e-2, Basemap error: None
def plot_set_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

# Chromaticity: 1.6e-2, Basemap error: None
def plot_binwise_chi_sq_bic_chromsmall_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: None
def plot_set_binwise_chrom_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: None
def plot_binwise_chi_sq_bic_chrom_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=None, savetag=savetag)

# Chromaticity: None (achromatic), Basemap error: 5
def plot_set_binwise_chrom0_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=5, savetag=savetag)

# Chromaticity: 0 (flat), Basemap error: 5
def plot_set_binwise_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=5)

# Chromaticity: 1.6e-2, Basemap error: 5
def plot_set_binwise_chromsmall_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

# Chromaticity: 1.6e-2, Basemap error: 5
def plot_binwise_chi_sq_bic_chromsmall_bm5(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 5
def plot_set_binwise_chrom_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=5, savetag=savetag)

# Chromaticity: 3.4e-2, Basemap error: 5
def plot_binwise_chi_sq_bic_chrom_bm5(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=5, savetag=savetag)
