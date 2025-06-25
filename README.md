# matrix-observer

Code used for the paper Ignatov & Pritchard 2025 [in review].

## Theory

We want to model the observations of a drift-scan dipole antenna in a CMB
mapmaking kind of way. For a binned timeseries data vector d, the model is

`d = GPYBa + n`

where `a` is a real vector of sky alm, `B` is a diagonal matrix of real beam 
alm, `Y` is a spherical harmonic matrix that maps between spherical harmonic 
space and pixel space, `P` is the pointing matrix, which encodes the observation 
strategy by placing a '1' in each row in the position corresponding to the pixel
being observed at a given time step. `G` is an averaging matrix that bins the 
time series data from length Nt to length `Ntau` < `Nt`. `n` is Gaussian random
noise. 

This has an analytic maximum-likelihood solution:

`W = inv( A.T inv(N) A ) A.T inv(N)`

`a_ml = W d`

where `W` is the maximum-likelihood estimator matrix.

21-cm antennas are insensitive to small angular scales, so a missing-mode
correction is employed. This corrects for the contributions of modes which
can't be estimated, but will bias the inference of the lower order modes. This
mean and covariance correction is computed with the `corr_2be` in the 
`sky_models.py` file. The stochastic foreground model used in Ignatov & Pritchard 2025 is
computed with `foreground_2be_alm_nsidelo` in `sky_models.py`.

## Setup

Run `setup.py`.

## Generating Paper Figures

`fig_gen.py` contains the functions to generate the figures in Ignatov & Pritchard 2025.

### Mapmaking Results

Run `gen_showcase_ml` to run the relevant inferences. Then, 
`plot_ml_showcase` will plot the pairwise figures, saved in 
`/fig/MLmod/pairplots/`, and `plot_showcase_ml_corner` will plot the corner
plot, saved in `/fig/MLmod/`.

### SSF Results

Run `gen_showcase_binwise` to run the relevant inferences. Then, 
`plot_showcase_binwise` will plot the 4-panel figure used in the paper, as well


## Modules

`multifrequency_ml_modelling.py`:
Implementation of the mapmaking method in the function 
`nontrivial_obs_memopt_missing_modes`. File contains running and plotter 
functions for a variety of cases, e.g. 

`run_set_gen_ml_chrom_bm5`, `plot_set_ml_chrom_bm5`, `plot_ml_chi_sq_bic_chrom_bm5`

for the chromaticity=3.4e-2, foreground correction 5% error case. 

`binwise_modelling.py`:
Implementation of the SSF method in the function `fg_cm21_chrom_corr`.
File contains running and plotter functions for a variety of cases, e.g. 

`run_set_gen_binwise_chromsmall_bm0`, `plot_binwise_chi_sq_bic_chromsmall_bm0`

for the chromaticity=1.6e-2 case, with zero basemap error.


In the folder `/src`:

`sky_models.py`:
Functions for calculating the **spherical harmonic coefficient vectors** of the 
foregrounds and 21-cm signal, and for adding noise to the timeseries data.

`beam_functions.py`:
Functions for calculating the **beam matrix** for an azimuthally symmetric beam.

`spherical_harmonics.py`:
Class for translating between real and complex spherical harmonic vectors and 
functions to calculate the **spherical harmonic matrix**.

`coordinates.py`:
Converting between different coordinate systems and functions to calculate the 
**pointing matrix**.

`forward_model.py`:
Functions that call the different matrix builder modules to build the total
**forward modelling matrix** for different preset observation strategies.

`map_making.py`:
Functions to calculate the **maximum-likelihood estimator matrix**.

`plotting.py`:
Useful plotting functions.

`blockmat.py`:
Classes to deal with the manipulation and operations of large block diagonal
matrices and vectors.


## Usage

### Single-frequency case

Manually forward-modelling observations of the GSMA sky up to a given `nside` 
and `lmax` at `nu` MHz:

```
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import src.sky_models as SM
import src.beam_functions as BM
import src.spherical_harmonics as SH
import src.coordinates as CO

# generate sky alm
alm_2be, map_2be = SM.foreground_2be_alm(nu, lmax, nside, map=True)

# visualise the fiducial sky
hp.mollview(map_2be, title="GSMA sky")
plt.show()

# construct the beam matrix
narrow_cosbeam = lambda theta : BM.beam_cos(theta, theta0=0.8)
mat_B = BM.calc_beam_matrix(nside, lmax, beam_use=narrow_cosbeam)

# construct the spherical harmonic matrix (can precompute for given lmax, nside)
mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)

# construct the observation matrix for multi-antenna drift-scan
coords_1 = CO.obs_zenith_drift_scan(lat_1, lon_1, 
            times=np.linspace(0, 24, 24, endpoint=False))
...
coords_n = CO.obs_zenith_drift_scan(lat_n, lon_n, 
            times=np.linspace(0, 24, 24, endpoint=False))
mat_P_multi_ant = CO.calc_pointing_matrix(coords_1, ..., coords_n, nside=32)

# optionally compute a binning matrix with Ntau bins
Nt = np.shape(mat_P_single_ant)[0]
mat_G = FM.calc_averaging_matrix(Ntau, Nt)

# compute the full observation matrix and forward model
mat_A = mat_G @ mat_P_single_ant @ mat_Y @ mat_B
timestream_data = mat_A @ alm_2be

# add noise for noise covariance matrix N in a 1 MHz frequency bin and t_int 
# hours of total integration time.
timestream_noisy, mat_N = SM.add_noise(timestream_data, dnu=1, Ntau=Ntau, 
            t_int=None, seed=123)
```


### BlockMat & BlockVec

Wanted code that - 
1. Is faster than using numpy as it doesn't store all the zeros involved.
2. Readily works for rectangular blocks (as long as the blocks are all the
same shape).

Can construct block matrices either by passing a list of identically-shaped 
blocks:

```
mat = [
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 2, 3],
    ],
    [
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
        [11, 12, 13],
    ], ...
]

block_mat = BlockMatrix(mat=mat)
```

then `block_mat.matrix` returns

```
[
    [1, 2, 3,  0,  0,  0, ...],
    [4, 5, 6,  0,  0,  0, ...],
    [7, 8, 9,  0,  0,  0, ...],
    [1, 2, 3,  0,  0,  0, ...],
    [0, 0, 0, 11, 12, 13, ...],
    [0, 0, 0, 14, 15, 16, ...],
    [0, 0, 0, 17, 18, 19, ...],
    [0, 0, 0, 11, 12, 13, ...],
    ...
]
```

or by passing a single block and specifying the number of desired repeats:

```
mat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3]
]

block_mat = BlockMatrix(mat=mat, nblock=12)
```

Can construct block vectors by passing:
1. A full block vector and specifying the number of blocks it contains.
2. A list of vector blocks.
3. A list of column vectors of blocks.

The full numpy representation of the vector is returned using 
`BlockVector.vector`, while a list of vector blocks is returned with 
`BlockVector.vector_blocks`.

The real purpose is of course to perform operations on these objects. This is 
done in the standard numpy-esque way, and is defined for vectors and matrices:

```
a@b
a+b
a-b
```

If the result is a vector, a BlockVector is returned. If the result is a matrix,
a BlockMatrix is returned.
