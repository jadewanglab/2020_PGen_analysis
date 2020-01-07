#%%
import pandas as pd
import numpy as onp
import jax.numpy as np

import pickle
import os

from numpyro.diagnostics import hpdi

import helpers as h
import plot_helpers as ph

# %%
direc = '.'
#%% read in samples from posterior for effect of conditions on gene expression
with open(os.path.join(direc,'big_horseshoe_model_samples.pkl'), 'rb') as pkl_file:
    samples = pickle.load(pkl_file)

#%% read in data
data = pd.read_csv(
    os.path.join(direc, 'data_long_with_design_info.csv.gz')
)

# %% wrangle the data to get some useful lookups
gene_info_df = data[~data.gene_lookup.duplicated()][['Name','Locus_tag','headon','gene_lookup']]
gene_info_df['Direction'] = gene_info_df.headon.replace({0:"Codirectional",1:"Head-on"})
cond_info_df = data[~data.condition_lookup.duplicated()][['condition','condition_lookup']]

gene_lookup = dict(
    zip(
        gene_info_df.Locus_tag.values,
        gene_info_df.gene_lookup.values
    )
)
gene_reverse_lookup = {val:k for k,val in gene_lookup.items()}
condition_lookup = dict(
    zip(
        cond_info_df.condition.values,
        cond_info_df.condition_lookup.values
    )
)
condition_reverse_lookup = {val:k for k,val in condition_lookup.items()}

locus_tags = data[~data.Locus_tag.duplicated()]['Locus_tag']
genes = data[~data.Locus_tag.duplicated()]['Name']
locus_tag_lookup = dict(zip(locus_tags, genes))
locus_tag_reverse_lookup = {v:k for k,v in locus_tag_lookup.items()}

# %% set LB exponential growth as intercept and get each conditions effect relative to LBexp baseline
LBexp_idx = condition_lookup['LBexp']

# calculate new samples for each gene's intercept (alpha) by adding LBexp beta to original alpha
new_alphas = samples['alpha'] + samples['b_condition'][:,:,LBexp_idx]
# calculate new betas by subtracting LBexp beta from each beta
new_betas = onp.zeros(samples['b_condition'].shape)
for i in range(new_betas.shape[2]):
    new_betas[:,:,i] = samples['b_condition'][:,:,i] - samples['b_condition'][:,:,LBexp_idx]

#%%
samples['new_alpha'] = new_alphas
samples['new_beta'] = new_betas

#%%
gene_trends_df = h.get_mean_and_ci(
    samples['new_beta'],
    key_name='condition_lookup',
    id_var='gene_lookup'
)
gene_trends_df.reset_index(inplace=True)

# %%
gene_trends_df['locus_tag'] = gene_trends_df.gene_lookup.replace(gene_reverse_lookup).values
gene_trends_df['gene'] = gene_trends_df.locus_tag.replace(locus_tag_lookup).values
gene_trends_df['condition'] = gene_trends_df.condition_lookup.replace(condition_reverse_lookup).values

#%%
gene_info_df['locus_tag'] = gene_info_df.Locus_tag
gene_trends_df = gene_trends_df.join(gene_info_df.drop(columns=["Name","Locus_tag","gene_lookup"]).set_index('locus_tag'), on='locus_tag')

#%%
for_gini = onp.zeros(samples['new_beta'].shape)
for i in range(for_gini.shape[0]):
    for_gini[i,...] = h.prep_for_gini(samples['new_beta'][i,...])

#%%
gini_arr = onp.zeros((for_gini.shape[0],for_gini.shape[1]))
for i in range(gini_arr.shape[0]):
    gini_arr[i,:] = h.gini(for_gini[i,:,:])

# %%
mean_gini = np.mean(gini_arr, axis=0)
gini_low,gini_up = hpdi(gini_arr, prob=0.9, axis=0)

gini_df = pd.DataFrame(
    {'mean_val':mean_gini,
     'lower_cl':gini_low,
     'upper_cl':gini_up,
     'locus_tag':[k for k in gene_lookup.keys()]}
)
gini_df['gene'] = gini_df.locus_tag.replace(locus_tag_lookup)
gini_df = gini_df.join(gene_info_df.set_index('locus_tag'), on='locus_tag').drop(columns='Name')
gini_df = gini_df.sort_values('mean_val')
gini_df['x_vals'] = onp.arange(gini_df.shape[0])

# %%
ph.plot_ginis(gini_df)

# %%
gini_density_plot = ph.plot_density(
    gini_df,
    data_var='mean_val',
    x_lab='Gini coefficient',
    y_lab='Density',
    color_var='Direction'
)
gini_density_plot

# %%
gini_density_plot.save('gini_distribution_horseshoe.png')
gini_density_plot.save('gini_distribution_horseshoe.svg')
