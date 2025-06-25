"""
Generate figures relevant for the paper.
"""
# Import the ROOT_DIR from the config file.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ROOT_DIR

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import healpy as hp
from matplotlib import cm, colorbar
from matplotlib.colors import Normalize

import src.observing as OBS
import src.sky_models as SM
import src.coordinates as CO
import src.beam_functions as BF
import src.forward_model as FM
import src.map_making as MM
from src.blockmat import BlockMatrix
from src.spherical_harmonics import RealSphericalHarmonics, calc_spherical_harmonic_matrix
from src.corner import AxesCornerPlot
from src.indices_2BE.generate_2BE_indices import T_CMB
RS = RealSphericalHarmonics()

import binwise_modelling as BM
import multifrequency_ml_modelling as MMM

alm2temp = 1/np.sqrt(4*np.pi)


################################################################################
# LMOD and NSIDE investigations.
################################################################################
def gen_lmod_investigation():
    def calc_d_vec(lmod=32, nside=64):
        npix    = hp.nside2npix(nside)
        lats  = OBS.ant_LUT[7]
        times = np.linspace(0, 24, 12, endpoint=False)
        nuarr   = np.array([70])
        narrow_cosbeam  = lambda x: BF.beam_cos_FWHM(x, FWHM=np.radians(60))
        
        # Generate foreground alm
        fg_alm_mod = SM.foreground_2be_alm_nsidelo(nu=nuarr, lmax=lmod, nside=nside, use_mat_Y=True)
        
        # Generate observation matrix for the modelling and for the observations.
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))

        # Calculate RMS errors of the data vectors.
        dmod = mat_A_mod@fg_alm_mod
        return dmod.vector
    pars = [2, 4, 8, 16, 32, 64]
    d_list = []
    for par in pars:
        d_list.append(calc_d_vec(par))
    np.save('INLR_d_list.npy', d_list)

def plot_lmod_nside_investigation():
    """
    Plot the RMS residuals to observing the sky from the LWA site in 3 time bins
    across a single day, compared across multiple LMOD values.
    This list is generated using NSIDE 64, with LMOD=[2, 4, 8, 16, 32, 64].
    The x axis ranges from LMOD=2->32, where e.g. the LMOD=2 point signifies the
    residuals between the LMOD=4 and LMOD=2 observations.

    Includes a second plot showing that the value of NSIDE=32 is enough to
    capture the behaviour of up to LMOD=64 modes, let alone LMOD=32 modes.
    """
    d_list = np.load('INLR_d_list.npy')
    pars = [2, 4, 8, 16, 32, 64]
    # Plot std error between each l value and the next l value, i.e. the first is RMS(l=2 - l=4).
    xx = list(range(len(d_list)-1))
    yy = [np.std(d_list[i]-d_list[i+1]) for i in range(len(d_list)-1)]

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    ax[0].loglog(pars[:-1],yy)
    ax[0].set_xticks(ticks=[], labels=[], minor=True)
    ax[0].set_xticks(ticks=pars[:-1], labels=pars[:-1], minor=False)
    ax[0].axhline(y=0.1, linestyle=':', color='k')
    ax[0].text(x=2.5,y=0.1*1.1, s="21-cm signal scale")
    ax[0].set_xlim(pars[0], pars[-2])
    ax[0].set_ylim(yy[-1], yy[0])
    ax[0].set_ylabel("RMS residual temperature [K]")
    ax[0].set_xlabel(r"$l_\mathrm{max}$")

    NSIDEs = [2, 4, 8, 16, 32, 64, 128]
    ELLs   = [32, 64]
    rads_NSIDE = [np.sqrt(4*np.pi / (12*NSIDE**2)) for NSIDE in NSIDEs]
    rads_ELL = [2*np.pi/(2*ELL) for ELL in ELLs]
    ax[1].loglog(NSIDEs, np.degrees(rads_NSIDE))
    sty = ['--', '-.']
    for ELL, rads, s in zip(ELLs, rads_ELL, sty):
        ax[1].axhline(y=np.degrees(rads), linestyle=s, color='k')
        ax[1].text(x=64, y=np.degrees(rads)*1.05, s="$l=$"+str(ELL), horizontalalignment='center')
    ax[1].set_xticks(ticks=[], labels=[], minor=True)
    ax[1].set_xticks(ticks=NSIDEs, labels=NSIDEs, minor=False)
    ax[1].set_xlim(NSIDEs[0], NSIDEs[-1])
    ax[1].set_ylim(np.degrees(rads_NSIDE[-1]), np.degrees(rads_NSIDE[0]))
    ax[1].set_xlabel("NSIDE")
    ax[1].set_ylabel("Approx pixel width [deg]")
    fig.tight_layout()
    plt.savefig(str(ROOT_DIR)+"/fig/lmod_nside_investigation.png")
    plt.savefig(str(ROOT_DIR)+"/fig/lmod_nside_investigation.pdf")
    plt.show()

def plot_lmod_investigation():
    """
    Plot the RMS residuals to observing the sky from the LWA site in 3 time bins
    across a single day, compared across multiple LMOD values.
    This list is generated using NSIDE 64, with LMOD=[2, 4, 8, 16, 32, 64].
    The x axis ranges from LMOD=2->32, where e.g. the LMOD=2 point signifies the
    residuals between the LMOD=4 and LMOD=2 observations.

    Does not include the second plot that plot_lmod_nside_investigation does.
    """
    d_list = np.load('INLR_d_list.npy')
    pars = [2, 4, 8, 16, 32, 64]
    # Plot std error between each l value and the next l value, i.e. the first is RMS(l=2 - l=4).
    xx = list(range(len(d_list)-1))
    yy = [np.std(d_list[i]-d_list[i+1]) for i in range(len(d_list)-1)]

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.loglog(pars[:-1],yy)
    ax.set_xticks(ticks=[], labels=[], minor=True)
    ax.set_xticks(ticks=pars[:-1], labels=pars[:-1], minor=False)
    ax.axhline(y=0.1, linestyle=':', color='k')
    ax.text(x=2.5,y=0.1*1.1, s="21-cm signal scale")
    ax.set_xlim(pars[0], pars[-2])
    ax.set_ylim(yy[-1], yy[0])
    ax.set_ylabel("RMS residual temperature [K]")
    ax.set_xlabel(r"$l_\mathrm{max}$")
    fig.tight_layout()
    fig.savefig(str(ROOT_DIR)+"/fig/lmod_investigation.png")
    fig.savefig(str(ROOT_DIR)+"/fig/lmod_investigation.pdf")
    plt.show()

################################################################################
# Skytrack maps with modelled and unmodelled foreground modes.
################################################################################
def plot_skytrack_maps():
    # Generate the sky tracks of 7 antennas.
    lats = [-3*26, -2*26, -1*26, 0, 1*26, 2*26, 3*26]
    coords = [CO.obs_zenith_drift_scan(lat, lon=0, times=np.linspace(0,24,1000)) for lat in lats]
    nside=256
    _,pix = CO.calc_pointing_matrix(*coords,nside=nside, pixels=True)
    m = np.zeros(hp.nside2npix(nside))
    m[pix] = 1
    hp.mollview(m)
    thetas, phis = hp.pix2ang(nside, pix)
    # Generate the foreground sky alm at 60 MHz.
    fg_alm = SM.foreground_2be_alm_nsidelo(nu=60, lmax=32, nside=32, use_mat_Y=True)

    # Generate a beam matrix to observe it with.
    mat_B  = BF.calc_beam_matrix(nside=32, lmax=32)

    # Generate/load the spherical harmonic matrix.
    mat_Y  = calc_spherical_harmonic_matrix(nside=32, lmax=32)

    # Split the spherical harmonic matrix into the low and high multipole sections, dividing at lmod.
    lmod  = 5
    Nlmod = RS.get_size(lmod)
    mat_Y_mod   = mat_Y[:,:Nlmod]
    mat_Y_unmod = mat_Y[:,Nlmod:]

    # Convolve the foregrounds with the beam.
    conv_fg = mat_B@fg_alm

    # Transform with the spherical harmonic matrix.
    mod_sky   = mat_Y_mod @ conv_fg[:Nlmod]
    unmod_sky = mat_Y_unmod @ conv_fg[Nlmod:]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,6))
    ls = (0, (8, 5))
    plt.axes(ax1)
    hp.mollview(mod_sky, hold=True, cbar=None, title=None)
    hp.projplot(thetas, phis, linestyle=ls, color='r', linewidth=2)

    plt.axes(ax2)
    hp.mollview(unmod_sky, hold=True, cbar=None, title=None)
    hp.projplot(thetas, phis, linestyle=ls, color='r', linewidth=2)

    normalize       = Normalize(vmin=np.min(mod_sky), vmax=np.max(mod_sky))
    scalar_mappable = cm.ScalarMappable(norm=normalize)
    colorbar_axis   = fig.add_axes([.16-0.06, .66-0.11, 0.03, .33])  # Colorbar location.
    cbar1 = colorbar.ColorbarBase(colorbar_axis, norm=normalize, 
                        orientation='vertical', ticklocation='left')
    cbar1.set_label(r'Temperature [K]')

    normalize       = Normalize(vmin=np.min(unmod_sky), vmax=np.max(unmod_sky))
    scalar_mappable = cm.ScalarMappable(norm=normalize)
    colorbar_axis   = fig.add_axes([.16-0.06, 0.13, 0.03, .33])  # Colorbar location.
    cbar2 = colorbar.ColorbarBase(colorbar_axis, norm=normalize, 
                        orientation='vertical', ticklocation='left')
    cbar2.set_label(r'Temperature [K]')
    plt.savefig(str(ROOT_DIR)+"/fig/skytrack_maps.pdf")
    plt.show()

################################################################################
# FWHM plot.
################################################################################
def plot_fwhm():
    """
    Plot the fwhm function from Tauscher et al. 2020 as a function of frequency
    for a range of values of the chromaticity parameter c.
    """
    nu = np.linspace(50, 100, 100)
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    c_values = [0, 1.6e-2, 3.4e-2, 5.2e-2]
    lss = [':', '--', '-.', (0, (6.4, 1.6, 1.0, 1.6, 1.0, 1.6))]
    cols = ["k", "C0", "C1", "C2"]
    for c, ls, col in zip(c_values, lss, cols):
        ax.plot(nu, np.degrees(BF.fwhm_func_tauscher(nu, c)), linestyle=ls, color=col)
    ax.axhline(y=72, linestyle='-', color='k')
    ax.text(x=60, y=72.2, s="achromatic")
    ax.text(x=62.9, y=65.8, s="0.0e-02", rotation=33)
    ax.text(x=65.8, y=62.5, s="1.6e-02", rotation=28)
    ax.text(x=69.5, y=58.1, s="3.4e-02", rotation=30)
    ax.text(x=73.6, y=54, s="5.2e-02", rotation=38)
    
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Beam FWHM [deg]")
    
    # Adjust x and y limits
    ax.set_xlim([50, 100])
    y_min = np.min([np.degrees(BF.fwhm_func_tauscher(nu, c)) for c in c_values])
    y_max = np.max([np.degrees(BF.fwhm_func_tauscher(nu, c)) for c in c_values])
    y_margin = (y_max - y_min) * 0.05  # Add a 5% margin to the y-axis limits
    ax.set_ylim([y_min - y_margin, y_max])
    
    fig.tight_layout()
    plt.savefig(str(ROOT_DIR)+"/fig/fwhm.pdf")
    plt.show()

################################################################################
# 
################################################################################
def plot_basemap_errs():
    def simp_basemap_err_to_delta(bmerr, ref_freq=70):
        return (bmerr/100)/np.log(408/ref_freq)
    def gaussian(x, sig, Nside=32):
        Npix = hp.nside2npix(Nside)
        return 2.35*Npix/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*x**2/sig**2)

    nuarr = OBS.nuarr
    delta_10 = simp_basemap_err_to_delta(10, ref_freq=70)
    delta_20 = simp_basemap_err_to_delta(20, ref_freq=70)

    delta_to_err = lambda delta: np.log(408/nuarr)*delta
    percentage_err_10 = delta_to_err(delta_10)
    percentage_err_20 = delta_to_err(delta_20)

    select_freqs = np.array([60,75,90])
    map_2be  = SM.foreground_2be_nsidelo(nu=select_freqs, nside=32)
    _, err_10 = SM.foreground_2be_alm_nsidelo(nu=select_freqs, lmax=32, nside=32, original_map=True, delta=delta_10)
    _, err_20 = SM.foreground_2be_alm_nsidelo(nu=select_freqs, lmax=32, nside=32, original_map=True, delta=delta_20)

    err_mean_10 = []
    for nu, map_2be_i in zip(select_freqs, map_2be):
        sigma_T   = delta_10 * np.log(408/nu)
        temp_mean_block = (map_2be_i - T_CMB) * np.exp(sigma_T**2/2) + T_CMB
        err_mean_10.append(temp_mean_block)
    
    err_mean_20 = []
    for nu, map_2be_i in zip(select_freqs, map_2be):
        sigma_T   = delta_20 * np.log(408/nu)
        temp_mean_block = (map_2be_i - T_CMB) * np.exp(sigma_T**2/2) + T_CMB
        err_mean_20.append(temp_mean_block)

    fig, ax = plt.subplots(1, 2, figsize=(6,2.8))
    ax[0].plot(nuarr, 1e2*percentage_err_10, label='10%')
    ax[0].plot(nuarr, 1e2*percentage_err_20, label='20%')
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("Fractional Std Dev [%]")
    ax[0].set_xlim(nuarr[0], nuarr[-1])
    ax[0].set_ylim(0, 1e2*np.max(percentage_err_20))

    bins = np.linspace(-50,50,41)
    ax[1].hist(1e2*(err_mean_20[0]-err_20[0])/err_mean_20[0], bins=bins, ec='k', alpha=0.5, color='C1', label=' 20%')
    ax[1].plot(bins, gaussian(bins, 20), color='C1',linewidth=1.5)
    ax[1].hist(1e2*(err_mean_10[0]-err_10[0])/err_mean_10[0], bins=bins, ec='k', alpha=0.5, color='C0', label=' 10%')
    ax[1].plot(bins, gaussian(bins, 10), color='C0',linewidth=1.5)
    ax[1].set_xlim(-50,50)
    ax[1].set_xlabel("Fractional Pixel Deviation [%]")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(str(ROOT_DIR)+"/fig/basemap_errs.pdf")
    plt.savefig(str(ROOT_DIR)+"/fig/basemap_errs.png")
    plt.show()

################################################################################
# Monopole reconstruction error figure.
################################################################################
def plot_monopole_reconstruction_err():
    """
    Showcase the monopole reconstruction error when we reconstruct only modes
    up to lmod for 2be truncated at lmod, and when we reconstruct only modes
    up to lmod for a non-truncated 2be.
    """
    # Generate single-frequency noisy foregrounds.
    fg = SM.foreground_2be_alm_nsidelo(nu=70, lmax=32, nside=32, use_mat_Y=True)

    # Truncate this at various ell values.
    ell_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    N_arr   = [RS.get_size(ell) for ell in ell_arr]
    fg_truncs = [fg[:N] for N in N_arr]

    def calc_mon_err(lats):
        # Generate observation matrix for a number of antennas.
        times = np.linspace(0, 24, 24, endpoint=False)
        narrow_cosbeam = lambda x: BF.beam_cos(x, 0.8)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside=32, lmax=32, 
                                                        lats=lats, 
                                                        times=times, beam_use=narrow_cosbeam, return_mat=True)

        # Observe for various ell values
        mat_A_truncs = [mat_A[:,:N] for N in N_arr]#[mat_P@mat_Y_i@mat_B_i for mat_Y_i,mat_B_i in zip(mat_Y_truncs, mat_B_truncs)]
        d_truncs = [mat_A_i@fg_i for mat_A_i,fg_i in zip(mat_A_truncs,fg_truncs)]

        # Add noise.
        d_noise_andcov_truncs = [SM.add_noise(d, dnu=1, Ntau=len(times), t_int=200, seed=456) for d in d_truncs]
        d_noise_truncs, d_cov_truncs = map(list, zip(*d_noise_andcov_truncs))
        print(f"noise mag for Nant {len(lats)} is {np.sqrt(np.mean([np.mean(d_cov) for d_cov in d_cov_truncs]))}")
        # Compute the maxlike estimator matrix for each case.
        mat_W_truncs = [MM.calc_ml_estimator_matrix(mat_A_i, mat_N_i, cond=True) for mat_A_i, mat_N_i in zip(mat_A_truncs, d_cov_truncs)]

        # Reconstruct the alm for each truncation case.
        alm_rec_truncs = [mat_W_i @ d_noise_i for mat_W_i,d_noise_i in zip(mat_W_truncs,d_noise_truncs)]
        alm_rec_truncs_nonoise = [mat_W_i @ d_i for mat_W_i,d_i in zip(mat_W_truncs,d_truncs)]

        # Visualise the reconstruction error for the monopole in each case.
        mon_err = [np.abs((alm_rec_i[0]-fg_i[0])*alm2temp) for fg_i,alm_rec_i in zip(fg_truncs,alm_rec_truncs)]
        mon_err_nonoise = [np.abs((alm_rec_i[0]-fg_i[0])*alm2temp) for fg_i,alm_rec_i in zip(fg_truncs,alm_rec_truncs_nonoise)]

        # Visualise the reconstruction error for the lmod=5 case.
        alm_err_lmod5 = np.abs((alm_rec_truncs[5]-fg_truncs[5])*alm2temp)

        return mon_err, mon_err_nonoise, alm_err_lmod5
    
    mon_err7, mon_err_nonoise7, alm_err_lmod5_7 = calc_mon_err([-3*26, -2*26, -1*26, 0, 1*26, 2*26, 3*26])
    mon_err5, mon_err_nonoise5, alm_err_lmod5_5 = calc_mon_err([-2*26, -1*26, 0, 1*26, 2*26])
    mon_err3, mon_err_nonoise3, alm_err_lmod5_3 = calc_mon_err([-1*26, 0, 26])

    
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    lss = [':', '--', '-.']
    ax[0].semilogy(ell_arr,mon_err_nonoise7, linestyle=lss[0], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err_nonoise5, linestyle=lss[1], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err_nonoise3, linestyle=lss[2], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err7, label=r'$N_\mathrm{ant}$=7', linestyle=lss[0])
    ax[0].plot(ell_arr,mon_err5, label=r'$N_\mathrm{ant}$=5', linestyle=lss[1])
    ax[0].plot(ell_arr,mon_err3, label=r'$N_\mathrm{ant}$=3', linestyle=lss[2])
    ax[0].axhline(y=0.1, color='k')
    ax0_majorticks = list(range(0,ell_arr[-1]+1,2))
    ax[0].set_xticks(ticks=ax0_majorticks, minor=False)
    ax[0].set_xticks(ticks=ell_arr, minor=True)
    ax[0].set_xlabel(r"$l_\mathrm{mod}$")
    ax[0].set_ylabel(r"Monopole Reconstruction Error [K]")
    ax[0].set_xlim(0,12)
    ax[0].set_ylim(0.004*1e-3, 1000*1e-3)
    ax[0].legend()

    ax[1].axhline(y=0.1, color='k')
    ax[1].semilogy(alm_err_lmod5_7, linestyle=lss[0])
    ax[1].semilogy(alm_err_lmod5_5, linestyle=lss[1])
    ax[1].semilogy(alm_err_lmod5_3, linestyle=lss[2])
    ax1_minorticks = list(range(0,len(alm_err_lmod5_7)))
    ax[1].set_xticks(ticks=ax1_minorticks, minor=True)
    ax[1].set_xlim(0,len(alm_err_lmod5_7)-1)
    ax[1].set_ylabel(r"Multipole Reconstruction Error [K]")
    ax[1].set_xlabel(r"$\mathbf{a}$ vector index")

    fig.tight_layout()
    plt.savefig(str(ROOT_DIR)+"/fig/monopole_reconstruction_err.pdf")
    plt.show()

################################################################################
# Chromatic and achromatic binwise modelling functions, with result plotting
# and chi-squared and BIC trend plotting.
################################################################################
def construct_runstr_ml(Nant, Npoly, chromstr, basemap_err):
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_idx<{basemap_err}>"
    elif basemap_err is None:
        runstr += f"_idx<0>"
    return runstr

def construct_runstr_bw(Nant, Npoly, chromstr, basemap_err):
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_idx<{basemap_err}>"
    return runstr

def gen_showcase_binwise():
    """Generate the binwise modelling results for the showcase figure."""
    BM.run_set_gen_binwise_chrom0_bm0(3,4,5,6,7)
    BM.run_set_gen_binwise_chromsmall_bm0(3,4,5,6,7)
    BM.run_set_gen_binwise_chrom_bm0(3,4,5,6,7)

def plot_showcase_binwise():
    """
    The final four-panel figure to showcase all binwise modelling/fitting.
    Involves a figure showing chrom0, chrom small, chrom large and the BIC plots
    for each of these all on one subplot.
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(6,6))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # Create subplots for first three panels (top-left, top-right, bottom-left)
    for i in range(3):
        row = i // 2
        col = i % 2
        # Create nested GridSpec for the panel with ratio 3:1
        nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, col],
                                                    height_ratios=[3, 1])
        ax_top = fig.add_subplot(nested_gs[0])
        ax_bottom = fig.add_subplot(nested_gs[1])
        
        # Store axes for later use
        if i == 0:
            ax_tl_top, ax_tl_bottom = ax_top, ax_bottom
        elif i == 1:
            ax_tr_top, ax_tr_bottom = ax_top, ax_bottom
        else:
            ax_bl_top, ax_bl_bottom = ax_top, ax_bottom

    # Create single subplot for bottom-right panel
    ax_br = fig.add_subplot(gs[1, 1])
    Npoly1, Npoly2, Npoly3 = 3, 5, 6
    runstr_tl = construct_runstr_bw(Nant=7, Npoly=Npoly1, chromstr=None, basemap_err=None)
    runstr_tr = construct_runstr_bw(Nant=7, Npoly=Npoly2, chromstr='1.6e-02', basemap_err=None)
    runstr_bl = construct_runstr_bw(Nant=7, Npoly=Npoly3, chromstr='3.4e-02', basemap_err=None)

    def construct_plot(ax_top, ax_bottom, runstr, Npoly):
        # Load data
        try:
            mcmcChain = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_mcmcChain.npy')
        except FileNotFoundError as e:
            raise FileNotFoundError("Cannot find MCMC chain for runstr. Make sure the run has been generated.") from e
        data      = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_data.npy')
        dataerr   = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_dataerr.npy')

        # Calculate contours and fid line.
        cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
        cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)
        idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
        samples_mcmcChain = mcmcChain[idx_mcmcChain]
        samples_mcmcChain = samples_mcmcChain[:,-3:]
        a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
        a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
        a00std_mcmc  = np.std(a00list_mcmc, axis=0)
        # Plot
        ax_top.plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
        ax_top.fill_between(
            OBS.nuarr,
            (a00mean_mcmc-a00std_mcmc)*alm2temp, 
            (a00mean_mcmc+a00std_mcmc)*alm2temp,
            color='C1',
            alpha=0.8,
            edgecolor='none',
            label="inferred"
        )
        ax_top.fill_between(
            OBS.nuarr,
            (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
            (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
            color='C1',
            alpha=0.4,
            edgecolor='none'
        )
        ax_top.set_ylabel("21-cm Temperature [K]")
        ax_top.set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
        ax_top.set_xticklabels([])
        ax_bottom.set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
        ax_bottom.set_xlabel("Frequency [MHz]")

        mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
        mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npoly)
        ax_bottom.axhline(y=0, linestyle=':', color='k')
        ax_bottom.errorbar(OBS.nuarr, mod(np.mean(mcmcChain, axis=0))-data, dataerr, fmt='.', color='k', ms=2)
        ax_bottom.set_ylabel(r"$T_\mathrm{res}$ [K]")

    construct_plot(ax_tl_top, ax_tl_bottom, runstr_tl, Npoly1)
    construct_plot(ax_tr_top, ax_tr_bottom, runstr_tr, Npoly2)
    construct_plot(ax_bl_top, ax_bl_bottom, runstr_bl, Npoly3)
    ax_tl_top.legend(loc='lower right')


    # Make bottom-right subplot.
    Npolys = [3,4,5,6,7]
    runstrs_chrom0     = [construct_runstr_bw(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=None) for Npoly in Npolys]
    runstrs_chromsmall = [construct_runstr_bw(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=None) for Npoly in Npolys]
    runstrs_chrom      = [construct_runstr_bw(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None) for Npoly in Npolys]
    bics_chrom0     = [np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chrom0]
    bics_chromsmall = [np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chromsmall]
    bics_chrom      = [np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chrom]
    ax_br.semilogy(Npolys, bics_chrom0, color='C0', linestyle='-', marker='o', label='achromatic')
    ax_br.semilogy(Npolys, bics_chromsmall, color='C1', linestyle='-', marker='s', label='c=1.6e-02')
    ax_br.semilogy(Npolys, bics_chrom, color='C2', linestyle='-', marker='^', label='c=3.4e-02')
    ax_br.set_ylabel("Model BIC")
    ax_br.set_xlabel("$N_\mathrm{poly}$")
    ax_br.legend(loc='upper right')
    ax_br.set_xticks(ticks=Npolys, labels=Npolys, minor=False)
    ax_br.set_xlim([Npolys[0], Npolys[-1]])

    # Set spacing between subplots
    fig.tight_layout()
    plt.savefig(str(ROOT_DIR)+"/fig/Binwise/showcase_binwise.pdf")
    plt.show()

    # CORNER PLOT.
    mcmcChain_tl  = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr_tl+'_mcmcChain.npy')
    mcmcChain_tr  = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr_tr+'_mcmcChain.npy')
    mcmcChain_bl  = np.load(str(ROOT_DIR)+'/saves/Binwise/'+runstr_bl+'_mcmcChain.npy')

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chain_tl = {tag: value for tag, value in zip(param_tags, mcmcChain_tl[:,-3:].transpose())}
    tagged_chain_tr = {tag: value for tag, value in zip(param_tags, mcmcChain_tr[:,-3:].transpose())}
    tagged_chain_bl = {tag: value for tag, value in zip(param_tags, mcmcChain_bl[:,-3:].transpose())}
    tagged_chain_tl['config'] = {'name' : ''}
    tagged_chain_tr['config'] = {'name' : ''}
    tagged_chain_bl['config'] = {'name' : ''}
    cornerplot = AxesCornerPlot(tagged_chain_bl, tagged_chain_tr, tagged_chain_tl, 
                                labels=param_labels, param_truths=OBS.cm21_params)
    cornerplot.set_figurepad(0.15)
    cornerfig = cornerplot.get_figure()
    cornerfig.savefig(str(ROOT_DIR)+"/fig/Binwise/showcase_binwise_corner.pdf")
    plt.show()


################################################################################
# Chromatic and achromatic ML modelling functions, with result plotting
# and chi-squared and BIC trend plotting.
################################################################################
def gen_showcase_ml():
    """Generate the ML modelling results for the showcase figure."""
    MMM.run_set_gen_ml_chrom0_bm0(3)
    MMM.run_set_gen_ml_chrom0_bm10(3)
    MMM.run_set_gen_ml_chromsmall_bm10(3)
    MMM.run_set_gen_ml_chrom_bm10(3)
    MMM.run_set_gen_ml_chromlarge_bm10(3)
    MMM.run_set_gen_ml_chrom_bm20(4)

def plot_ml_showcase():
    """
    The set of three pairwise plots to showcase all ML modelling/fitting.
    """
    plot_ml_chrom_pair(Nant1=7, Nant2=7, Npoly1=3, Npoly2=3, chromstr1=None, chromstr2=None, basemap_err1=None, basemap_err2=10, savetag="")
    plot_ml_chrom_pair(Nant1=7, Nant2=7, Npoly1=3, Npoly2=3, chromstr1='1.6e-02', chromstr2='3.4e-02', basemap_err1=10, basemap_err2=10, savetag="")
    plot_ml_chrom_pair(Nant1=7, Nant2=7, Npoly1=3, Npoly2=4, chromstr1='5.2e-02', chromstr2='3.4e-02', basemap_err1=10, basemap_err2=20, savetag="")


def plot_ml_chrom_pair(Nant1=7, Nant2=7, Npoly1=7, Npoly2=7, chromstr1=None, chromstr2=None, basemap_err1=None, basemap_err2=None, savetag=None):
    runstr1 = construct_runstr_ml(Nant1, Npoly1, chromstr1, basemap_err1)
    runstr2 = construct_runstr_ml(Nant2, Npoly2, chromstr2, basemap_err2)
    print("loading from", runstr1, "and", runstr2, sep='\n')

    fig, ax = plt.subplots(2, 2, figsize=(6,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    try:
        mcmcChain = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr1+'_mcmcChain.npy')
    except FileNotFoundError as e:
        raise FileNotFoundError("Cannot find MCMC chain for runstr1. Make sure the run has been generated.") from e
    residuals = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr1+'_modres.npy')
    data      = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr1+'_data.npy')
    dataerr   = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr1+'_dataerr.npy')
    rec_a00   = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr1+"_rec_a00.npy")
    a00_error = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr1+"_rec_a00_err.npy")
    
    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant1))
    data  = np.reshape(data, (Nfreq, Nant1, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant1, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant1, Ntau))

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)
    ax[0,0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0,0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1,0].set_xlabel("Frequency [MHz]")
    ax[0,0].set_ylabel(r"21-cm Temperature [K]")
    ax[0,0].legend()
    top_plot_spacing = 0.02
    ax00_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax00_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    ax[1,0].axhline(y=0, linestyle=':', color='k')
    ax[1,0].errorbar(OBS.nuarr, (rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k', ms=2)
    ax[1,0].axhline(0, linestyle=':', color='k')
    ax[1,0].set_ylabel(r"$\hat{T}_\mathrm{mon}-\mathcal{M}$ [K]")
    bottom_plot_spacing = 0.01
    ax10_ymax = np.max(rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp + np.max(a00_error*alm2temp) + bottom_plot_spacing
    ax10_ymin = np.min(rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp - np.max(a00_error*alm2temp) - bottom_plot_spacing

    try:
        mcmcChain = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr2+'_mcmcChain.npy')
    except FileNotFoundError as e:
        raise FileNotFoundError("Cannot find MCMC chain for runstr2. Make sure the run has been generated.") from e
    residuals = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr2+'_modres.npy')
    data      = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr2+'_data.npy')
    dataerr   = np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr2+'_dataerr.npy')
    rec_a00   = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr2+"_rec_a00.npy")
    a00_error = np.load(str(ROOT_DIR)+"/saves/MLmod/"+runstr2+"_rec_a00_err.npy")

    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant2))
    data  = np.reshape(data, (Nfreq, Nant2, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant2, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant2, Ntau))

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)
    ax[0,1].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0,1].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1,1].set_xlabel("Frequency [MHz]")
    ax01_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax01_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    ax[1,1].axhline(y=0, linestyle=':', color='k')
    ax[1,1].errorbar(OBS.nuarr, (rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k', ms=2)
    ax[1,1].axhline(0, linestyle=':', color='k')
    ax11_ymax = np.max(rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp + np.max(a00_error*alm2temp) + bottom_plot_spacing
    ax11_ymin = np.min(rec_a00-MMM.fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp - np.max(a00_error*alm2temp) - bottom_plot_spacing

    ax[0,0].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[0,1].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[1,0].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])
    ax[1,1].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])
    # Turn off the y axis ticklabels for the right plots.
    ax[0,1].set_yticklabels([])
    ax[1,1].set_yticklabels([])

    fig.tight_layout()
    if savetag is not None:
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/pairplots/ml_"+runstr1+"and"+runstr2+savetag+".pdf")
        plt.savefig(str(ROOT_DIR)+"/fig/MLmod/pairplots/ml_"+runstr1+"and"+runstr2+savetag+".png")
    plt.show()


def plot_showcase_ml_corner():
    """
    Final corner plot for showcasing ML modelling, featuring the 21-cm
    posteriors for achromatic case and c=3.4e-2 both with 10% foreground 
    correction errors, as well as c=3.4e-2 with 20% foreground correction 
    errors.
    """
    Nants = [7,7,7]
    Npolys = [3,3,4]
    chromstrs = [None, '3.4e-02', '3.4e-02']
    basemap_errs = [10, 10, 20]
    runstrs = [construct_runstr_ml(Nants[i], Npolys[i], chromstrs[i], basemap_errs[i]) for i in range(3)]
    print("loading from", runstrs)

    mcmcChains = [np.load(str(ROOT_DIR)+'/saves/MLmod/'+runstr+'_mcmcChain.npy') for runstr in runstrs]
    cm21_chains = [chain[:,-3:] for chain in mcmcChains]

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chains = [{tag: value for tag, value in zip(param_tags, chain.transpose())} for chain in cm21_chains]
    for i in range(3):
        tagged_chains[i]['config'] = {'name' : '', 'shade_alpha' : 0.1}
    
    cornerplot = AxesCornerPlot(tagged_chains[2], tagged_chains[1], tagged_chains[0], 
                                labels=param_labels, param_truths=OBS.cm21_params,
                                plotter_kwargs={'figsize':"COLUMN"})
    cornerplot.set_figurepad(0.15)
    f = cornerplot.get_figure()
    f.savefig(str(ROOT_DIR)+"/fig/MLmod/showcase_ml_corner.pdf")
    plt.show()


