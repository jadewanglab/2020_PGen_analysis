#%%
import pandas as pd
import numpy as onp
from scipy import stats
from pprint import pprint
from Bio import SeqIO
import os

import helpers as h

# %%
direc = '.'

#%% read in regulations from subtiwiki
regulations = pd.read_csv(os.path.join(direc, 'Subtiwiki_regulations.csv'))
regulations['locus_tag'] = regulations['locus tag'].str.replace('_','')

#%%
# read in B. subtilis 168 genome
refseq = SeqIO.read(
    os.path.join(direc, 'B_subtilis_168_NC_000964.3.gbk'),
    'genbank',
)
# get CDSs from 168 genome
CDS = [feat for feat in refseq.features if feat.type == "CDS"]

#%% build data frame from CDS list
locus_tags = []
names = []
products = []
starts = []
ends = []
strands = []
for feat in CDS:
    info = feat.qualifiers
    locus_tags.append(info['old_locus_tag'][0])
    if 'gene' in info:
        names.append(info['gene'][0])
    else: names.append('')
    products.append(info['product'][0])
    starts.append(int(feat.location.start))
    ends.append(int(feat.location.end))
    strand = feat.location.strand
    if strand == 1:
        strand = '+'
    else: strand = '-'
    strands.append(strand)

# now make the dataframe
cds_df = pd.DataFrame(
    data = {
        'locus_tag': locus_tags,
        'gene': names,
        'product': products,
        'start': starts,
        'end': ends,
        'strand': strands
    }
)

# %% identify which genes are head-on, which are co-directional
rtp_end = int(cds_df[cds_df.gene == 'rtp'].end)

cds_df['headon'] = 0
cds_df.loc[(cds_df['start'] > rtp_end) & (cds_df['strand'] == '+'), 'headon'] = 1
cds_df.loc[(cds_df['start'] <= rtp_end) & (cds_df['strand'] == '-'), 'headon'] = 1

#%% get specific regulons from subtiwiki info
regulon_dict = {
    'sigB' : regulations[regulations.regulator=="SigB"]['locus_tag'],
    'sigV' : regulations[regulations.regulator=="SigV"]['locus_tag'],
    'sigM' : regulations[regulations.regulator=="SigM"]['locus_tag'],
    'sigX' : regulations[regulations.regulator=="SigX"]['locus_tag'],
    'sigY' : regulations[regulations.regulator=="SigY"]['locus_tag'],
    'sinR' : regulations[regulations.regulator=="SinR"]['locus_tag'],
    # only use genes activated by spx, to limit analysis to those genes upregulated during spx-regulated stress response
    'spx' : regulations[(regulations.regulator=="Spx") & (regulations['mode'] == "activation")]['locus_tag'],
    'abrB' : regulations[regulations.regulator=="AbrB"]['locus_tag'],
    'spo0A' : regulations[regulations.regulator=="Spo0A"]['locus_tag'],
}

# %% annotate CDSs with information on whether each CDS is in each regulon
for regulon, locus_tags in regulon_dict.items():
    cds_df[regulon] = 0
    cds_df.loc[cds_df.locus_tag.isin(locus_tags),regulon] = 1

#%% check results of above code
regulon_contents = {}
for regulon in regulon_dict.keys():
    regulon_contents[regulon] = {}

    regulon_contents[regulon]['cds_df'] = len(onp.where(cds_df[regulon] == 1)[0])
    regulon_contents[regulon]['subtiwiki'] = len(regulon_dict[regulon])

pprint(regulon_contents)
# result of above code:
    # sigB is missing 20 from cds_df
    # sigV is good
    # sigM is missing 12 from cds_df
    # sinR is good
    # spx is missing 2 from cds_df
    # abrB is missing 8 from cds_df
    # spo0A is missing 26 from cds_df

# %%
missing_idx = {}
for regulon,wiki_vals in regulon_dict.items():

    cds_vals = cds_df[cds_df[regulon] == 1]['locus_tag']
    missing_idx[regulon] = onp.where(~wiki_vals.isin(cds_vals))[0]

pprint(missing_idx)

# %%
for regulon,idx in missing_idx.items():
    if len(idx) > 0:
        missing_loci = regulon_dict[regulon].iloc[idx,]
        print(regulon)
        print(missing_loci)

# missing loci are misc_RNA or other. Not CDSs.
# addition missing in the numbers above are duplicate records

#%%
headon_regulon_dict = {}
for regulon in regulon_dict.keys():
    headon_regulon_dict[regulon] = cds_df.groupby(
        ['headon', regulon]
    ).gene.count()

#%%
headon_regulon_obs = {}
headon_regulon_exp = {}
headon_regulon_pvals = {}
for regulon,counts in headon_regulon_dict.items():

    headon = counts[1]
    # check whether head-on genes have and zero counts
    # categories with zero counts will be absent,
    #   so supplement here with zero.
    if not 0 in headon:
        headon[0] = 0
    if not 1 in headon:
        headon[1] = 0

    # check whether co-directional genes have and zero counts
    # categories with zero counts will be absent,
    #   so supplement here with zero.
    codir = counts[0]
    if not 0 in codir:
        codir[0] = 0
    if not 1 in codir:
        codir[1] = 0

    # make 2x2 contingency table
    contingency_table = onp.array(
        [headon,
        codir]
    )
    headon_regulon_obs[regulon] = contingency_table

    # run Chi-square test
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    headon_regulon_exp[regulon] = ex
    headon_regulon_pvals[regulon] = p

# %%
for regulon in headon_regulon_obs:
    print(regulon)
    pprint(headon_regulon_obs[regulon])
    pprint(headon_regulon_exp[regulon])

# %%
pprint(headon_regulon_pvals)

# %%
