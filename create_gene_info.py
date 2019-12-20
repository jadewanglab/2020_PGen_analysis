#%%
import pandas as pd

# %%
direc = '.'
data = pd.read_csv(os.path.join(direc, 'data_long.csv.gz'))

#%%
gene_lookup = dict(
    zip(
        data.Locus_tag.unique(),
        range(len(data.Locus_tag.unique()))
    )
)

condition_lookup = dict(
    zip(
        data.condition.unique(),
        range(len(data.condition.unique()))
    )
)

# %%
data['gene_lookup'] = data.Locus_tag.replace(gene_lookup).values
data['condition_lookup'] = data.condition.replace(condition_lookup).values

#%%
data.to_csv(
    'data_long_with_design_info.csv.gz',
    index=False,
    columns=["Name","Locus_tag","headon","condition","replicate","log2signal","gene_lookup","condition_lookup"]
)

# %%
