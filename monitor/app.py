from dash import Dash, html, dcc, Input, Output, callback
import dash
from fuvtds_base_class import FUVTDSBase, FUVTDSMonitor
from plotly.subplots import make_subplots
from astropy.time import Time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from io import StringIO
from astropy.convolution import Box1DKernel, convolve

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)



# i hope this works
def generate_data(PIDs='fuvtds_analysis_list.dat'):
    fuvtds = FUVTDSBase(PIDs=PIDs)
    df = fuvtds.tables
    TDSDates = fuvtds.TDSDates

    return(df, TDSDates)

# Current TDSTAB in use
TDSTAB = '/grp/hst/cdbs/lref/83j20454l_tds.fits'


#-------------------------------------------------------------------------------
#### App Layout ####
#-------------------------------------------------------------------------------
# Define the layout
app.layout = html.Div(children=[
    dcc.Tabs(
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children = [
        dcc.Tab(label='Relative Sensitivity', className='custom-tab', selected_className='custom-tab--selected', children=[
            # Container for the overall layout
            html.Div(children=[
                # Container for the left side (Dropdowns)
                html.Div(children=[
                    html.Button("Run Analysis", id="run-button", style={
                        'width': '150px',
                        'height': '50px',
                        'fontSize': '10px'
                    }),
                    dcc.Loading(
                        id="loading",
                        type="default",
                    ),
                    dcc.Store(id="computed-results"), # store for the computed results,
                    dcc.Store(id="dates"), # store for the computed results,
                    html.Br(),

                    html.Label('Grating'),
                    dcc.Dropdown(
                        id='gratings'),
                    html.Br(),

                    html.Label('Cenwave'),
                    dcc.Dropdown(
                        id='cenwaves'
                    ),
                    html.Br(),

                    html.Label('Segment'),
                    dcc.Dropdown(id='segments'),
                    html.Br(),

                    html.Label('Size'),
                    dcc.Dropdown(id='sizes'),

                    html.Div(id='display-selected-values')
                ], style={'width': '10%', 'display': 'inline-block'}),

                # Container for the plot in the middle
                html.Div(children=[
                    dcc.Graph(id='relative-net')
                ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top'}),

                # Container for the right side (RadioItems)
                html.Div(children=[
                    html.Label('Wavelength Bins'),
                    dcc.RadioItems(
                        id='wavelength-bins'
                    ),
                    html.Br()
                ], style={'max-height': '300px', 'overflow-y': 'scroll', 'padding': '10px'})
            ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),  # Flexbox for layout
      ]),

        # need to play around with the sizes of this of the solar plot
      dcc.Tab(label='Solar Flux', className='custom-tab', selected_className='custom-tab--selected', children=[
          # solar flux plot
          dcc.Graph(id='solar-flux')  
      ]),


      dcc.Tab(label='Slope vs Time', className='custom-tab', selected_className='custom-tab--selected', children=[
          dcc.Graph(id='time-slope')
        ])  
    ])
])

## ------- generate data ------- ##
@callback(
        Output("computed-results", "data"),
        Output("dates", "data"),
        Input("run-button", "n_clicks"),
)
def run_computation(n_clicks):
    if n_clicks is None:
        return dash.no_update
    # run the computation
    df, monitor = generate_data()
    df_json = df.to_json(date_format='iso', orient='split')
    return df_json, monitor
## ------- generate data ------- ##


## ------- PLOTS PLOTS PLOTS ------- ##
## ------- PLOTS PLOTS PLOTS ------- ##
@callback(
       Output('relative-net', 'figure'),
       Input('computed-results', 'data'),
       Input('dates', 'data'),
       Input('gratings', 'value'),
       Input('cenwaves', 'value'),
       Input('segments', 'value'),
       Input('sizes', 'value'),
       Input('wavelength-bins', 'value'))
def update_rel_sens_graph(data, dates, selected_grating, selected_cenwave, selected_segment, selected_size, selected_wl_bin):
    if data is None:
        fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'Relative net count rate',
                                        ''))
        return fig
    df = pd.read_json(StringIO(data), orient='split')

    df = df[
        (df['opt_elem'] == selected_grating) & 
        (df['cenwave'] == selected_cenwave) & 
        (df['segment'] == selected_segment)]
    
    date = np.array(df['date-obs']).flatten()

    monitor = FUVTDSMonitor(dates)

    scaled_df = monitor.scale_to_1(table=df, size=selected_size) # scale to one

    net  = np.array([net for net in scaled_df[f'{selected_size}_scaled_net'].iloc[0]])
    net_err = np.array([net for net in scaled_df[f'{selected_size}_scaled_stdev'].iloc[0]])
    best_fit_model = np.array([net for net in scaled_df[f'{selected_size}_best_fit'].iloc[0]])

    binned_wl = np.array([wl for wl in df[f'{selected_size}_binned_wl']])[0]
    wl_edges  = np.array([wl for wl in df[f'{selected_size}_wl_edges']])[0]

    # TDS model
    tds = np.transpose([1.0/monitor.tds_backout(
        1.0, binned_wl,
        Time(mjd, format='decimalyear').mjd,
        selected_grating, 'ANY',
        selected_segment, selected_cenwave, TDSTAB
    ) for mjd in date])

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{selected_grating}/{selected_cenwave}/{selected_segment} - Relative net count rate',
                                        ''))

    for i, wl in enumerate(binned_wl):
        if wl == selected_wl_bin:
            curr_sens = go.Scatter(
                x=date,
                y=net[:,i],
                error_y = dict(type='data', array=net_err[:,i], visible=True),
                mode='markers',
                name=f"Current TDS {wl_edges[i]} - {wl_edges[i+1]}",
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['date-obs-fits']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'Date Obs: %{customdata[4]}'
                "<extra></extra>",
                line_color='black'
            )

            tds_model = go.Scatter(
                x = date,
                y = tds[i],
                name = 'Current TDSTAB',
                line = dict(color='orange', width=4, dash='dash')
            )
            x = np.append(monitor.reftime, monitor.breakpoints)
            x = np.append(x, date[-1])
            best_fit = go.Scatter(
                x = x,
                y =monitor.broken_lines(x - monitor.reftime, *best_fit_model[i,:]),
                name = 'Best Fit',
                line = dict(color='grey', width=4, dash='dash')
            )

            # plot two!# plot two!# plot two!# plot two!
            ymodel = tds[i]
            residual = 100.0*(net[:,i] - ymodel) / ymodel
            sens_err = go.Scatter(
                x = date,
                y = residual,
                mode='markers',
                name='TDSTAB',
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['date-obs-fits']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'Date Obs: %{customdata[4]}'
                "<extra></extra>",
                line_color='orange'
            )

            ymodel = monitor.broken_lines(date - monitor.reftime, *best_fit_model[i,:])
            residual = 100.0*(net[:,i] - ymodel) / ymodel
            best_fit_err = go.Scatter(
                x = date,
                y = residual,
                mode='markers',
                name='Best Fit',
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['date-obs-fits']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'Date Obs: %{customdata[4]}'
                "<extra></extra>",
                line_color='black'
            )



            # Fill between
            fig.add_trace(go.Scatter(
                x = [x[0]-0.1, x[-1]+0.1],
                y = [-5, -5],
                mode='lines',
                fill = 'tonexty',
                showlegend=False,
                line_color='teal'), row=2, col=1)
            fig.add_trace(go.Scatter(
                x = [x[0]-0.1, x[-1]+0.1],
                y = [5, 5],
                mode='lines',
                fill = 'tonexty',
                showlegend=False,
                line_color='teal'), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x = [x[0]-0.1, x[-1]+0.1],
                y = [-2, -2],
                mode='lines',
                fill = 'tozeroy',
                showlegend=False,
                line_color='purple'), row=2, col=1)
            fig.add_trace(go.Scatter(
                x = [x[0]-0.1, x[-1]+0.1],
                y = [2, 2],
                mode='lines',
                fill = 'tozeroy',
                showlegend=False,
                line_color='purple'), row=2, col=1)
            

            # add traces
            fig.add_trace(curr_sens, row=1, col=1)
            fig.add_trace(best_fit, row=1, col=1)
            fig.add_trace(tds_model, row=1, col=1)


            # add vertical lines here
            fig.add_traces(monitor.add_lines(monitor.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
            fig.add_traces(monitor.add_lines(monitor.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint', [min(residual)-3, max(residual)+3],True), rows=2, cols=1)
            fig.add_traces(monitor.add_lines(monitor.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
            fig.add_traces(monitor.add_lines(monitor.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch', [min(residual)-3, max(residual)+3], True), rows=2, cols=1)

            if selected_segment == 'FUVA':
                fig.add_traces(monitor.add_lines(monitor.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
                fig.add_traces(monitor.add_lines(monitor.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA',[min(residual)-3, max(residual)+3], True), rows=2, cols=1)
            else:
                fig.add_traces(monitor.add_lines(monitor.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
                fig.add_traces(monitor.add_lines(monitor.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB', [min(residual)-3, max(residual)+3], True), rows=2, cols=1)
            
            # add more traces
            fig.add_trace(sens_err, row=2, col=1)
            fig.add_trace(best_fit_err, row=2, col=1)

            # update the y-axis based on what mode is being plotted
            fig.update_yaxes(range=(min(net[:,i])-0.2, max(net[:,i])+0.2), row=1, col=1),
            fig.update_yaxes(range=(min(residual)-3, max(residual)+3), row=2, col=1)

    # update the x-axes
    fig.update_xaxes(title_text='Date', range=(x[0]-0.1, x[-1]+0.1), row=2, col=1)
    fig.update_yaxes(title_text='Relative net count rate', row=1, col=1),
    fig.update_yaxes(title_text='Percent Difference', row=2, col=1)

    fig.update_layout(height=700)

    return (fig)


## ------- PLOTS PLOTS PLOTS ------- ##
@callback(
       Output('solar-flux', 'figure'),
       Input('computed-results', 'data'),
       Input('dates', 'data'))
def update_solar_flux_graph(data, dates):
    if data is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        return fig
    
    # !!! grab necessary data !!!
    # read in the fuv tds data as a dataframe
    df = pd.read_json(StringIO(data), orient='split')

    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)

    # get the solar flux data as a dataframe
    solar = monitor.get_solar_data()


    # !!! Plot the solar flux from dataframe, both smoothed and unsmoothed
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    unsmoothed = go.Scatter(x = solar['date'],
                            y = solar['f10.7'],
                            line_shape='linear',
                            line=dict(color='royalblue', width=4),
                            name='10.7 cm radio flux',
                            opacity=0.5)
    smoothed = go.Scatter(x = solar['date'],
                            y = convolve(solar['f10.7'], Box1DKernel(150), boundary='extend'),
                            line_shape='linear',
                            line=dict(color='firebrick', width=2),
                            name='Smoothed 10.7 cm radio flux',
                            opacity=0.6)
    
    # differentiate the A seg from B seg
    marker_type = {'FUVA': 'circle', 'FUVB': 'x'}

    # plot the fractional throughput
    for cenwave in df['cenwave'].unique():
        for segment in df['segment'][df['cenwave'] == cenwave].unique():

            sub_df = df[
                (df['cenwave'] == cenwave) & 
                (df['segment'] == segment)]
            # scale to one for large bins (always been large bins)
            scaled_df = monitor.scale_to_1(table=sub_df, size='large')

            grating = sub_df['opt_elem'].iloc[0]
            date = np.array(sub_df['date-obs']).flatten()
            net = np.array([net for net in scaled_df['large_scaled_net'].iloc[0]])

            # Fractional throughput, loop over all monitored modes
            frac_throughput = go.Scatter(
                x = date,
                y = net[:,0],
                mode='markers',
                marker_symbol=marker_type[segment],
                name=f'{grating}/{cenwave}/{segment}',
                customdata = np.stack(
                    (sub_df['rootname'],
                     sub_df['life_adj'],
                     sub_df['proposid'],
                     sub_df['targname']),
                     axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}'
                "<extra></extra>"
            )

            # add trace to plot
            fig.add_trace(frac_throughput, secondary_y=False)
    
    # add solar flux smoothed and unsmoothed to plot based off secondary_y
    fig.add_trace(unsmoothed, secondary_y=True)
    fig.add_trace(smoothed, secondary_y=True)

    # add vertical lines
    fig.add_traces(monitor.add_lines(monitor.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint', [0, 1.2]))
    fig.add_traces(monitor.add_lines(monitor.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch', [0, 1.2]))
    fig.add_traces(monitor.add_lines(monitor.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA', [0, 1.2]))
    fig.add_traces(monitor.add_lines(monitor.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB', [0, 1.2]))

    # add title
    fig.update_layout(title_text="TDS Solar Flux", height=700)

    # set x-axis title
    fig.update_xaxes(title_text="Date", range=(min(solar['date']), max(solar['date'])))

    # set y-axis titles
    fig.update_yaxes(title_text="Fractional Throughput", range=(0.0, 1.1), secondary_y=False)
    fig.update_yaxes(title_text="10.7 cm Flux (units here)", range=(50, 400), secondary_y=True)
    
    return (fig)

## ------- PLOTS PLOTS PLOTS ------- ##
@callback(
       Output('time-slope', 'figure'),
       Input('computed-results', 'data'),
       Input('dates', 'data'))
def update_solar_flux_graph(data, dates):
    if data is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        return fig




## ------- PLOTS PLOTS PLOTS ------- ##
## ------- PLOTS PLOTS PLOTS ------- ##





## ------- BUTTONS ------- ##
## ------- BUTTONS ------- ##
# Gratings
@callback(
    Output('gratings', 'options'),
    [Input('computed-results', 'data')])
def set_grating_options(data):
    if data is None:
        return "No results available."
    df = pd.read_json(StringIO(data), orient='split')
    return [{'label': i, 'value': i} for i in df['opt_elem'].unique()]
@callback(
    Output('gratings', 'value'),
    [Input('gratings', 'options')])
def set_grating_value(available_options):
    return available_options[0]['value']

# Cenwaves
@callback(
    Output('cenwaves', 'options'),
    [Input('computed-results', 'data'),
     Input('gratings', 'value')])
def set_cenwave_options(data, selected_grating):
    if data is None:
        return "No results available."
    df = pd.read_json(StringIO(data), orient='split')
    return [{'label': i, 'value': i} for i in df['cenwave'][df['opt_elem'] == selected_grating].unique()]
@callback(
    Output('cenwaves', 'value'),
    [Input('cenwaves', 'options')])
def set_cenwave_value(available_options):
    return available_options[0]['value']

# Segments
@callback(
    Output('segments', 'options'),
    [Input('computed-results', 'data'),
     Input('gratings', 'value'),
     Input('cenwaves', 'value')])
def set_segment_options(data, selected_grating, selected_cenwave):
    if data is None:
        return "No results available."
    df = pd.read_json(StringIO(data), orient='split')
    return [{'label': i, 'value': i} for i in df['segment'][(df['opt_elem'] == selected_grating) & (df['cenwave'] == selected_cenwave)].unique()]
@callback(
    Output('segments', 'value'),
    [Input('segments', 'options')])
def set_segment_value(available_options):
    return available_options[0]['value']

# Sizes
@callback(
    Output('sizes', 'options'),
    [Input('computed-results', 'data')])
def set_size_options(data):
    if data is None:
        return "No results available."
    return [{'label': 'Small', 'value': 'small'}, {'label': 'Large', 'value': 'large'}]
@callback(
    Output('sizes', 'value'),
    [Input('sizes', 'options')])
def set_size_value(available_options):
    return available_options[0]['value']

# Wavelength bins
@callback(
    [Output('wavelength-bins', 'options'),
     Output('wavelength-bins', 'value')],
    [Input('computed-results', 'data'),
     Input('gratings', 'value'),
     Input('cenwaves', 'value'),
     Input('segments', 'value'),
     Input('sizes', 'value')])
def set_wavelength_options(data, selected_grating, selected_cenwave, selected_segment, selected_size):
    if data is None:
        return "No results available."
    df = pd.read_json(StringIO(data), orient='split')

    binned_wl = np.array([wl for wl in df[f'{selected_size}_binned_wl'][(df['opt_elem'] == selected_grating) &
                                                             (df['cenwave'] == selected_cenwave) &
                                                             (df['segment'] == selected_segment)]])
    wl_edges  = np.array([wl for wl in df[f'{selected_size}_wl_edges'][(df['opt_elem'] == selected_grating) &
                                                            (df['cenwave'] == selected_cenwave) &
                                                            (df['segment'] == selected_segment)]])

    for i, wl in enumerate(binned_wl):
        options = []
        for j, _ in enumerate(wl):
            options.append({'label': f'{wl_edges[i][j]} - {wl_edges[i][j+1]}',
                            'value': binned_wl[i][j]})
        value = binned_wl[i][0]
        return(options, value)
## ------- BUTTONS ------- ##
## ------- BUTTONS ------- ##


if __name__ == '__main__':
    app.run(debug=True)