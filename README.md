# Analysis of data from Nicolas et al., Science 2012

## Introduction

This repository contains data and code to reproduce analysis described in
Schroeder, Sankar, Wang and Simmons, PLoS Genet, 2020.

Some of the python scripts referenced in this readme rely on the included python files `helpers.py` and `plot_helpers.py` to be read in as modules.

## Data preparation

Table S2 from (Nicolas, P. et al. Condition-dependent transcriptome reveals high-level regulatory architecture in Bacillus subtilis. Science 335, 1103–1106 (2012)) was downloaded from the following url: http://genome.jouy.inra.fr/basysbio/bsubtranscriptome/. Table S2 was saved as "TableS2_Nicolas_et_al.csv"

Data contained in "TableS2_Nicolas_et_al.csv" were tidied into a long format using the code in `nicolas_analysis.R`. Output of running the code in `nicolas_analysis.R` was saved in "data_long.csv.gz".

Data were further curated using code in `create_gene_info.py`. Running `create_gene_info.py` generated the file "data_long_with_design_info.csv.gz".

"data_long_with_design_info.csv.gz", contains data with the following variables:

| Name | Locus_tag | headon | condition | replicate | log2signal | gene_lookup | condition_lookup |

Variable names correspond to the following meanings:

1. Name: common gene name
2. Locus_tag: feature locus tag
3. headon: direction of transcription relative to replication, where 0 is codirectional and 1 is head-on
4. condition: experimental condition from Nicolas et al., Science 2012.
5. replicate: replicate number for the given condition.
6. log2signal: quoted from the legend of Nicolas et al., Science 2012, Table S2; "median of the estimated transcription signal at probes within the feature"
7. gene_lookup: integer; unique identifier for each locus tag
8. condition_lookup: integer; unique identifier for each of the 104 conditions tested in Nicolas et al., Science 2012.

## Analysis of effect of conditions from Nicolas et al., Science 2012 on gene expression

Bayesian analysis of each condition's effect on each gene's expression was performed using the python module [numpyro](https://github.com/pyro-ppl/numpyro.git). I ran the following code from the linux terminal:

```bash
python3 ./big_horseshoe_model_fit_script.py
```

The python script `big_horseshoe_model_fit_script.py` samples the posterior distribution for, among other parameters, each gene's intercept log2signal and the effect of each condition on the log2signal for each gene. A Finnish Horseshoe prior (Piironen and Vehtari, Electron J Stat 2017) was applied to each gene to avoid inferring many false-positive effects.

Running `big_horseshoe_model_fit_script.py` generates the file `big_horseshoe_model_samples.pkl`. Due to the large size of `big_horseshoe_model_samples.pkl`, I could not include it in thie repository. However, running the code as described will recreate my results.

## Interpretation of sampled posteriors

The sampled posteriors from `big_horseshoe_model_fit_script.py` were then interpreted using code in `analysis.py`.

## Testing for enrichment of head-on genes in the SigB regulon

Table 1 from Schroeder, Sankar, Wang and Simmons, PLoS Genet, 2020 was prepared using code in `analysis.py`. Annotations of genes from the Nicolas et al. dataset as in/not in the SigB regulon were taken from the file `Subtiwiki_regulations.csv`, contains information on known regulatory interactions in _B. subtilis_. `Subtiwiki_regulations.csv` is included in the repository for reproducibility, or can be downloaded (the information may have changed since our analysis) at http://subtiwiki.uni-goettingen.de/v3/exports.
