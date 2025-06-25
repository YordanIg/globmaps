"""
Define likelihoods, priors, etc.
"""
import numpy as np
from emcee import EnsembleSampler
from scipy.optimize import curve_fit


def log_likelihood(theta, y, yerr, model):
    """
    Compute the Gaussian log-likelihood, given a model(theta) and data y
    with errors yerr.
    """
    y_model = model(theta)
    chi2 = (y - y_model)**2 / (yerr**2)
    return np.sum(-chi2 / 2)

def log_likelihood_vectors(theta, y, invcov, model):
    """
    Compute the Gaussian log-likelihood, given a model(theta) and data y
    with inverse covariance matrix invcov.
    """
    y_model = model(theta)
    chi2 = (y - y_model) @ invcov @ (y - y_model)
    return np.sum(-chi2 / 2)

def log_prior(theta, prior_pars):
    """
    Log Prior probability

        log(prior(theta))

        This handles an unnormalised uniform prior
        within the rectangular bounds given by prior_pars.

        inputs:
        theta - N array of parameter values
        prior_pars - [N,2] array of prior ranges
        i.e. = [[lower1, upper1], ...]

        Returns 0 if theta in prior_pars and -inf otherwise
    """
    lower =  theta > prior_pars[:,0]
    upper = theta < prior_pars[:,1]
    in_prior_pars = all(lower & upper)

    #return prior value
    if in_prior_pars:
        return 0.0
    return -np.inf

def log_posterior(theta, y, yerr, model, prior_pars):
    lp = log_prior(theta, prior_pars)
    if np.isfinite(lp):
        lp += log_likelihood(theta, y, yerr, model)
    return lp

def log_posterior_vectors(theta, y, invcov, model, prior_pars):
    lp = log_prior(theta, prior_pars)
    if np.isfinite(lp):
        lp += log_likelihood_vectors(theta, y, invcov, model)
    return lp

def prior_checker(priors, p0):
    """
    Check that each value of the vector p0 lies within the prior bounds of
    priors. If not, replace that value of the vector p0 with the mean of the
    prior bound passed.
    NOTE: You must not pass any prior bounds that average to zero.
    """
    lower =  p0 > priors[:,0]
    upper = p0 < priors[:,1]
    in_prior_pars = lower & upper
    for i in range(len(p0)):
        if not in_prior_pars[i]:
            print(f"ERROR: {i}th parameter of p0 not in prior volume. Setting to mean of prior.")
            p0[i] = np.mean(priors[i])
    return p0

def curve_fit_emcee(f, xdata, ydata, sigma, bounds, p0=None, chain=False):
    """
    An EMCEE version of the scipy curve_fit function. Can return the MCMC chain
    if chain=True.

    If p0 is None, will run the standard curve_fit function to get the initial
    guess for the parameters.
    """
    steps=10000
    burn_in=3000
    def mod(theta):
        return f(xdata, *theta)
    if p0 is not None:
        assert len(p0) == len(bounds[0])
        assert len(p0) == len(bounds[1])
    
    if p0 is None:
        cf_p0 = np.mean(bounds, axis=0)
        p0, _ = curve_fit(f, xdata, ydata, sigma=sigma, bounds=bounds, p0=cf_p0)
        print("Initial guess from curve_fit:", p0)
    nwalkers = 64
    ndim = len(p0)
    pos = p0*(1 + 1e-4*np.random.randn(nwalkers, ndim))

    priors = np.array((list(zip(bounds[0], bounds[1]))))
    p0 = prior_checker(priors, p0)
    sampler = EnsembleSampler(nwalkers, ndim, log_posterior, 
                        args=(ydata, sigma, mod, priors))
    _=sampler.run_mcmc(pos, nsteps=steps, progress=True, skip_initial_state_check=True)
    chain_mcmc = sampler.get_chain(flat=True, discard=burn_in)
    if chain:
        return np.mean(chain_mcmc, axis=0), np.cov(chain_mcmc, rowvar=False), chain_mcmc
    return np.mean(chain_mcmc, axis=0), np.cov(chain_mcmc, rowvar=False)
