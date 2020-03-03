#%%
import numpy as onp

import jax.random as random

import pandas as pd
import pickle
import os

import helpers as h

# %%
direc = '.'
data = pd.read_csv(os.path.join(direc, 'data_long_with_design_info.csv.gz'))

#%% check number of y-vals for each gene, important later for horseshoe prior
y_val_count = []
for gid in onp.unique(data.gene_lookup.values):
    y_val_count.append(data[data.gene_lookup==gid].shape[0])

# %%
N = onp.unique(y_val_count)[0] # all genes have 269 y-values

#%%
# Start from this source of randomness.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

num_warmup, num_samples = 1000,500
num_chains = 1

#%%
data_dict = {
    'y_vals':data.log2signal.values,
    'gid':data.gene_lookup.values,
    'cid':data.condition_lookup.values,
    'N':N, # number of y-vals for each gene, calculated above,
    'condition_intercept':True
}

samples = h.sample_model(
    rng_key,
    model=h.horseshoe_model,
    model_args_dict=data_dict,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains
)

#%%
# no need to keep these horseshoe priors, just keep alpha and bC
del samples['beta_tilde']
del samples['lambd']
del samples['tau_tilde']
del samples['c2_tilde']

#%%
with open('big_horseshoe_model_condition_intercept_samples.pkl','wb') as pkl_file:
    pickle.dump(samples, pkl_file)