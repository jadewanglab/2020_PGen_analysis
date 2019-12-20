import altair as alt
import seaborn as sns
import numpy as np


# %%
def plot_trace(samples,
               data,
               gene,
               condition=None,
               gene_lut=None,
               condition_lut=None,
               locus_tag_reverse_lut=None,
               param_name='b_condition'):
    
    try: #assume we have locus_tag, if not, lookup tag from gene name
        gene_idx = gene_lut[gene]
    except KeyError:
        loc_tag = locus_tag_reverse_lut[gene]
        gene_idx = gene_lut[loc_tag]

    arr = samples[param_name]
    if len(arr.shape) > 2:
        condition_idx = condition_lut[condition]
        these_samples = samples[param_name][:,gene_idx,condition_idx]
    else:
        these_samples = samples[param_name][:,gene_idx]

    sns.lineplot(x=np.arange(these_samples.size), y=these_samples)

# %%
def plot_effects(df, gene=None, locus_tag=None):

    if gene is not None:
        if 'Name' in df.columns:
            df = df[df.Name == gene].copy()
        elif 'gene' in df.columns:
            df = df[df.gene == gene].copy()
    if locus_tag is not None:
        df = df[df.locus_tag == locus_tag].copy()

    df = df.sort_values(['mean_val'])
    df['x_vals'] = np.arange(df.shape[0])
    
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('x_vals', title='Condition'),
        y=alt.Y('mean_val', title='Effect of condition (x)'),
        tooltip='condition'
    )

    band = alt.Chart(df).mark_area().encode(
        x=alt.X('x_vals', title='Condition'),
        y=alt.Y('lower_cl', title='Effect of condition (x)'),
        y2=alt.Y2('upper_cl', title='Effect of condition (x)'),
        opacity=alt.value(0.35),
        tooltip='condition'
    )

    if gene is not None:
        plot = (line + band).properties(
            title = gene
        )
    elif locus_tag is not None:
        plot = (line + band).properties(
            title = locus_tag
        )
    else:
        plot = (line + band)

    return(plot)

# %%
def set_base_density_chart(base_chart,
                           data_var,
                           color_var=None,
                           extent=None):

    if color_var is None:
        density_base = base_chart.transform_density(
            data_var,
            as_=[data_var,'density'],
            # extent=extent
        )
    else:
        density_base = base_chart.transform_density(
            data_var,
            as_=[data_var,'density'],
            groupby=[color_var],
            # extent=extent
        )

    return(density_base)

def make_mean_rule(base_chart, other_chart, data_var, color_var=None):

    if color_var is None:
        rule_chart = base_chart.mark_rule().encode(
            x=alt.X(
                'mean({})'.format(data_var),
                scale=alt.Scale(zero=False)
            ),
            size=alt.value(3)
        )

    else:
        rule_chart = base_chart.mark_rule().encode(
            x=alt.X(
                'mean({})'.format(data_var),
                scale=alt.Scale(zero=False)
            ),
            size=alt.value(3),
            color='{}:N'.format(color_var)
        )

    plot = rule_chart + other_chart

    return(plot)

def encode_density(base_chart,
                   data_var,
                   color_var,
                   x_lab,
                   y_lab,
                   stack):
                #    extent):

    density_base = set_base_density_chart(
        base_chart,
        data_var,
        color_var,
        # extent
    )

    density_plot = density_base.mark_area(
        opacity=0.35,
    ).encode(
        alt.X(
            '{}:Q'.format(data_var),
            title=x_lab
        ),
        alt.Y(
            'density:Q',
            title=y_lab,
            stack=stack
        )
    )

    return(density_plot)

def plot_density(
    df,
    data_var,
    condition=None,
    x_lab='',
    y_lab='',
    color_var=None,
    stack=False,
    include_mean=True,
    facet_var=None
    ):

    if condition is not None:
        if type(condition) == str:
            df = df[df.condition == condition]
        else:
            df = df[df.condition.isin(condition)]

    base_chart = alt.Chart(df)

    density_chart = encode_density(
        base_chart,
        data_var,
        color_var,
        x_lab,
        y_lab,
        stack
    )

    if include_mean:
        plot = make_mean_rule(
            base_chart,
            density_chart,
            data_var,
            color_var
        )
    else:
        plot = density_chart

    if color_var is not None:
        plot = plot.encode(color='{}:N'.format(color_var))

    if facet_var is not None:
        plot = plot.facet(row='{}:N'.format(facet_var))
        plot = plot.configure_legend(
            strokeColor='gray',
            fillColor='white',
            padding=10,
            cornerRadius=10,
            orient='none',
            legendX=14,
            legendY=0.14,
            symbolType='square'
        )
    else:
        plot = plot.configure_legend(
            strokeColor='gray',
            fillColor='white',
            padding=10,
            cornerRadius=10,
            orient='top-right',
            symbolType='square'
        )

    plot = plot.configure_axis(
        labelFontSize=15,
        titleFontSize=15
    )

    return(plot)

def plot_gene_density_for_condition(df, condition=None, color_var=None, stack=True):

    if condition is not None:
        df = df[df.condition==condition]

    genes_base = set_base_density_chart(df,color_var)

    genes_bar = genes_base.mark_area(
        opacity=0.34,
        # interpolate='step'
    ).encode(
        alt.X(
            'mean_val',
            bin=alt.Bin(maxbins=25),
            title='Effect of {}'.format(condition),
        ),
        alt.Y(
            'count()',
            title='Total number of genes in bin',
            stack=stack
        )
    )

    genes_rule = genes_base.mark_rule().encode(
        x=alt.X(
            'mean(mean_val)',
            scale=alt.Scale(zero=False)
        ),
        size=alt.value(3)
    )

    if color_var is not None:
        genes_plot = (genes_rule + genes_bar).encode(color='{}:N'.format(color_var))
    else:
        genes_plot = genes_rule + genes_bar

    return(genes_plot)

def plot_gini_hist(df, color_var=None, stack=True):

    base = alt.Chart(df)
    bar = base.mark_area(
        opacity=1,
        interpolate='step'
    ).encode(
        alt.X(
            'mean_val',
            bin=alt.Bin(maxbins=25),
            title='Gini coefficient',
        ),
        alt.Y(
            'count()',
            title='Total number of genes in bin',
            stack=stack
        )
    )

    rule = base.mark_rule().encode(
        x=alt.X(
            'mean(mean_val)',
            scale=alt.Scale(zero=False)
        ),
        size=alt.value(3)
    )

    if color_var is not None:
        plot = (rule + bar).encode(color='{}:N'.format(color_var))
    else:
        plot = rule + bar

    return plot

def plot_genes(df, condition="LBexp", color_var=None):

    df = df[df.condition==condition]
    df = df.sort_values(['mean_val'])
    df['x_vals'] = np.arange(df.shape[0])
    
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('x_vals', title='Condition'),
        y=alt.Y('mean_val', title='Effect of condition (x)'),
        tooltip='gene'
    )

    band = alt.Chart(df).mark_area().encode(
        x=alt.X('x_vals', title='Condition'),
        y=alt.Y('lower_cl', title='Effect of condition (x)'),
        y2=alt.Y2('upper_cl', title='Effect of condition (x)'),
        opacity=alt.value(0.35),
        tooltip='gene'
    )

    if color_var is not None:
        plot = (line + band).encode(
            color='{}:N'.format(color_var)
        )
    else:
        plot = (line + band)

    return(plot)

def plot_ginis(df, color_var=None, facet_var=None):

    line = alt.Chart(df).mark_line().encode(
        x=alt.X('x_vals', title='Gene'),
        y=alt.Y('mean_val', title='Gini'),
        color='{}:N'.format(color_var),
        tooltip='gene'
    )

    band = alt.Chart(df).mark_area().encode(
        x=alt.X('x_vals', title='Gene'),
        y=alt.Y('lower_cl', title='Gini'),
        y2=alt.Y2('upper_cl', title='Gini'),
        opacity=alt.value(0.35),
        fill='{}:N'.format(color_var),
        tooltip='gene'
    )

    if color_var is not None:
        plot = (line + band).encode(
            color='{}:N'.format(color_var)
        )
    else:
        plot = line + band

    if facet_var is not None:
        plot = plot.facet(
            row='{}:N'.format(facet_var)
        )

    return(plot)