import altair as alt
import numpy as onp

import jax
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi

import pandas as pd

#%%
def get_mean_and_ci(x, key_name, id_var, prob=0.9, axis=0):
    mean_val = x.mean(axis=axis)
    low_ci,up_ci = hpdi(x, prob=prob, axis=axis)

    df = pd.DataFrame(mean_val)
    df[id_var] = df.index.values

    df_mean_long = pd.melt(
        df,
        var_name=key_name,
        value_name='mean_val',
        id_vars=id_var
    )

    multi_idx = pd.MultiIndex.from_frame(df_mean_long[[id_var,key_name]])
    df_mean_long.index = multi_idx
    df_mean_long.drop(
        columns=[key_name,id_var],
        inplace=True
    )

    df = pd.DataFrame(low_ci)
    df[id_var] = df.index.values

    df_low_long = pd.melt(
        df,
        var_name=key_name,
        value_name='lower_cl',
        id_vars=id_var
    )

    df_low_long.index = multi_idx
    df_low_long.drop(
        columns=[key_name,id_var],
        inplace=True
    )

    df = pd.DataFrame(up_ci)
    df[id_var] = df.index.values

    df_up_long = pd.melt(
        df,
        var_name=key_name,
        value_name='upper_cl',
        id_vars=id_var
    )

    df_up_long.index = multi_idx
    df_up_long.drop(
        columns=[key_name,id_var],
        inplace=True
    )

    df = df_mean_long.join(df_low_long).join(df_up_long)

    return df

def subtract_min(x):
    y = x - np.amin(x)
    return(y)

def add_a_bit(x):
    y = x + 0.0000001 #values for Gini calculation cannot be 0
    return(y)

def prep_for_gini(x):
    y = subtract_min(x)
    z = add_a_bit(y)
    return(z)

prep_for_gini = jax.jit(jax.vmap(prep_for_gini, in_axes=0, out_axes=0))

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

gini = jax.jit(jax.vmap(gini, in_axes=0, out_axes=0))

#%% set up for finnish horseshoe in (Piironen, J., and Vehtari, A., 2017)
def finnish_horseshoe(M, m0, N, var, half_slab_df, slab_scale2, tau_tilde, c2_tilde, lambd, beta_tilde):

    tau0 = (m0/(M-m0) * (np.sqrt(var)/np.sqrt(1.0*N)))
    tau = tau0 * tau_tilde
    c2 = slab_scale2 * c2_tilde
    lambd_tilde = np.sqrt(c2 * lambd**2 / (c2 + tau**2 + lambd**2))

    beta = tau * lambd_tilde * beta_tilde

    return(beta)

def horseshoe_model(y_vals,
                    gid,
                    cid,
                    N, # array of number of y_vals in each gene
                    slab_df=1,
                    slab_scale=1,
                    expected_large_covar_num=5): # expected large covar num here is the prior on the number of conditions we expect to affect expression of a given gene

    gene_count = gid.max()+1
    condition_count = cid.max()+1

    # separate regularizing prior on intercept for each gene
    a_prior = dist.Normal(10., 10.)
    a = numpyro.sample("alpha", a_prior, sample_shape=(gene_count,))

    # implement Finnish horseshoe
    half_slab_df = slab_df/2
    variance = y_vals.var()
    slab_scale2 = slab_scale**2
    hs_shape = (gene_count, condition_count)

    # set up "local" horseshoe priors for each gene and condition
    beta_tilde = numpyro.sample('beta_tilde', dist.Normal(0., 1.), sample_shape=hs_shape) # beta_tilde contains betas for all hs parameters
    lambd = numpyro.sample('lambd', dist.HalfCauchy(1.), sample_shape=hs_shape) # lambd contains lambda for each hs covariate
    # set up global hyperpriors.
    # each gene gets its own hyperprior for regularization of large effects to keep the sampling from wandering unfettered from 0.
    tau_tilde = numpyro.sample('tau_tilde', dist.HalfCauchy(1.), sample_shape=(gene_count,1))
    c2_tilde = numpyro.sample('c2_tilde', dist.InverseGamma(half_slab_df, half_slab_df), sample_shape=(gene_count,1))

    bC = finnish_horseshoe(M = hs_shape[1], # total number of conditions
                            m0 = expected_large_covar_num, # number of condition we expect to affect expression of a given gene
                            N = N, # number of observations for the gene
                            var = variance,
                            half_slab_df = half_slab_df,
                            slab_scale2 = slab_scale2,
                            tau_tilde = tau_tilde,
                            c2_tilde = c2_tilde,
                            lambd = lambd,
                            beta_tilde = beta_tilde)
    numpyro.sample("b_condition", dist.Delta(bC), obs=bC)
    
    # calculate implied log2(signal) for each gene/condition
    #   by adding each gene's intercept (a) to each of that gene's
    #   condition effects (bC).
    mu = a[gid] + bC[gid,cid]

    sig_prior = dist.Exponential(1.)
    sigma = numpyro.sample('sigma', sig_prior)
    return numpyro.sample('obs', dist.Normal(mu, sigma), obs=y_vals)

# %%
def sample_model(rng_key,
                 model,
                 data,
                 gid,
                 cid,
                 N,
                 num_warmup=500,
                 num_samples=500,
                 num_chains=1):

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_warmup,
        num_samples,
        num_chains,
        progress_bar=True
    )

    mcmc.run(
        rng_key,
        y_vals=data,
        gid=gid,
        cid=cid,
        N=N
    )

    mcmc.print_summary()
             
    # divergences = mcmc.get_extra_fields()["diverging"]

    samples = mcmc.get_samples()
    # samples['divergences'] = divergences

    bC = numpyro.infer.Predictive(
            model,
            samples
        ).get_samples(
            rng_key,
            y_vals=data,
            gid=gid,
            cid=cid,
            N=N
        )

    samples['b_condition'] = bC

    return samples
