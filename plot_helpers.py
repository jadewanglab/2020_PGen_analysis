import altair as alt


def set_base_density_chart(base_chart,
                           data_var,
                           color_var=None,
                           extent=None):

    if color_var is None:
        density_base = base_chart.transform_density(
            data_var,
            as_=[data_var,'density'],
        )
    else:
        density_base = base_chart.transform_density(
            data_var,
            as_=[data_var,'density'],
            groupby=[color_var],
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

    density_base = set_base_density_chart(
        base_chart,
        data_var,
        color_var,
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