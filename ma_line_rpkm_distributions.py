#%%
import pandas as pd
import plot_helpers as ph

# %%
direc = '.'
ma_data = pd.read_csv(os.path.join(direc, 'PY79_ma_line_z-scores.csv'))

# %%
ma_data['Direction'] = ma_data.withReplication.replace({"codirectional":"Codirectional","head-on":"Head-on"})

# %%
ma_plot = ph.plot_density(
    ma_data,
    data_var='logRPKMstd',
    x_lab='z-score of ln(RPKM)',
    y_lab='Density',
    color_var='Direction',
    include_mean=True
)
ma_plot

# %%
ma_plot.save(os.path.join(direc,'PY79_logRPKMstd_density.png'))
ma_plot.save(os.path.join(direc,'PY79_logRPKMstd_density.svg'))

