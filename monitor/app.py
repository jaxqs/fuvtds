from dash import Dash, html, dcc, Input, Output, callback
import dash
from fuvtds_base_class import FUVTDSBase, FUVTDSMonitor
from plotly.subplots import make_subplots
from astropy.time import Time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from io import StringIO, BytesIO
from astropy.convolution import Box1DKernel, convolve
import zipfile

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# this generates the data needed to run monitor
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
    # Container for the labels, dropdowns, and button (on top of the tabs)
    html.Div(children=[
        # Grating Dropdown
        html.Div(children=[
            html.Label('Grating', style={
                'display': 'block',
                'textAlign': 'center',
            }),
            dcc.Dropdown(
                id='gratings', style={
                    'display': 'block',
                    'width': '80%',
                    'margin': '0 auto'
                }
            ),
        ], style={'width': '20%', 'textAlign': 'center'}),

        # Cenwave Dropdown
        html.Div(children=[
            html.Label('Cenwave', style={
                'display': 'block',
                'textAlign': 'center',
            }),
            dcc.Dropdown(
                id='cenwaves', style={
                    'display': 'block',
                    'width': '80%',
                    'margin': '0 auto'
                }
            ),
        ], style={'width': '20%', 'textAlign': 'center'}),  

        # Segment Dropdown
        html.Div(children=[
            html.Label('Segment', style={
                'display': 'block',
                'textAlign': 'center',
            }),
            dcc.Dropdown(
                id='segments', style={
                    'display': 'block',
                    'width': '80%',
                    'margin': '0 auto'
                }
            ),
        ], style={'width': '20%', 'textAlign': 'center'}),  

        # Size Dropdown
        html.Div(children=[
            html.Label('Size', style={
                'display': 'block',
                'textAlign': 'center',
            }),
            dcc.Dropdown(
                id='sizes', style={
                    'display': 'block',
                    'width': '80%',
                    'margin': '0 auto'
                }
            ),
        ], style={'width': '20%', 'textAlign': 'center'}),

        # Run Button
        html.Div(children=[
            html.Button("Run Analysis", id="run-button", style={
                'width': '100%',
                'height': '50px',
                'fontSize': '10px',
            }),
            dcc.Loading(id="loading", type="default"),
            dcc.Store(id="computed-results"),
            dcc.Store(id="dates"),
        ], style={'width': '20%', 'textAlign': 'center'}),
        
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),

    # Space between the dropdowns and the tabs
    html.Br(),

    # Tabs container
    dcc.Tabs(
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Relative Sensitivity',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
                    html.Div(children=[
                        # Container for the plot in the middle
                        html.Div(children=[
                            html.Button("Download Throughput", id='btn_csv'),
                            dcc.Download(id='large-thruput'),
                            dcc.Graph(id='relative-net', style={'height': '700px'})
                        ], style={'width': '85%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Container for the right side (RadioItems)
                        html.Div(children=[
                            html.Button("Save as HTML", id='save-relative-net'),
                            dcc.Download(id='relative-html'),
                            html.Label('Wavelength Bins'),
                            dcc.RadioItems(
                                id='wavelength-bins'
                            ),
                            html.Br()
                        ], style={'max-height': '300px', 'overflow-y': 'scroll', 'padding': '10px'})
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),  # Flexbox for layout
                ]
            ),
            dcc.Tab(
                label='Solar Flux',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
                    html.Button("Save as HTML", id='save-solar-flux'),
                    dcc.Download(id='solar-flux-html'),
                    dcc.Graph(id='solar-flux', style={'height': '700px'})
                ]
            ),
            dcc.Tab(
                label='Slope vs Time',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
                    html.Div(children=[
                        # Container for the plot in the middle
                        html.Div(children=[
                            html.Button("Save as HTML", id='save-time-slope'),
                            dcc.Download('time-slope-html'),
                            dcc.Graph(id='time-slope', style={'height': '700px'})
                        ], style={'width': '85%', 'display': 'inline-block', 'vertical-align': 'top'}),

                        # Container for the right side (RadioItems)
                        html.Div(children=[
                            html.Label('Time bins'),
                            dcc.RadioItems(
                                id='timebins'
                            ),
                            html.Br()
                        ], style={'max-height': '300px', 'overflow-y': 'scroll', 'padding': '10px'})
                    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),  # Flexbox for layout
                ]
            )
        ]
    )
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

            fig, x = monitor.rel_sens_graph(date, net, net_err, i, wl_edges, best_fit_model, df, tds, fig)
    # update the x-axes
    fig.update_xaxes(title_text='Date', range=(x[0]-0.1, x[-1]+0.1), row=2, col=1)
    fig.update_yaxes(title_text='Relative net count rate', row=1, col=1),
    fig.update_yaxes(title_text='Percent Difference', row=2, col=1)

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

    return fig

## ------- PLOTS PLOTS PLOTS ------- ##
@callback(
       Output('time-slope', 'figure'),
       Input('computed-results', 'data'),
       Input('dates', 'data'),
       Input('sizes', 'value'),
       Input('timebins', 'value'))
def update_time_slope_graph(data, dates, selected_size, selected_time_bin):

    if data is None:
        fig = go.Figure()
        return fig
    
    # !!! grab necessary data !!!
    # read in the fuv tds data as a dataframe
    df = pd.read_json(StringIO(data), orient='split')

    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)

    # Establish plot
    fig = go.Figure()

    fig = monitor.time_slope_graph(df, selected_size, selected_time_bin, fig)

    return fig
## ------- PLOTS PLOTS PLOTS ------- ##
## ------- PLOTS PLOTS PLOTS ------- ##



## ------- SAVING METHODS ------- ##
## ------- SAVING METHODS ------- ##
# Saving Large Throughput -- as csv
@callback(
        Output('large-thruput', 'data'),
        Input('btn_csv', 'n_clicks'),
        Input('computed-results', 'data'),
        Input('dates', 'data'))
def save_large_thruput(n_clicks, data, dates):
    if n_clicks is None:
        return dash.no_update
    if data is None:
        print('run analysis')
    
    df = pd.read_json(StringIO(data), orient='split')

    tables = []

    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)
    for cenwave in df['cenwave'].unique():
        for segment in df['segment'][df['cenwave'] == cenwave].unique():
            sub_df = df[(df['cenwave'] == cenwave) & (df['segment'] == segment)]
            scaled_df = monitor.scale_to_1(table=sub_df, size='large') # scale to one

            # This with statement is to remove the false positivie error in doing this process
            with pd.option_context("mode.copy_on_write", True):
                # change scaled net count to a format we can use
                sub_df.loc[:,'relative_throughput'] = np.array([net for net in scaled_df.loc[0, 'large_scaled_net']])
                sub_df.loc[:,'relative_throughput_err'] = np.array([net for net in scaled_df.loc[0,'large_scaled_stdev']])

            # combine the relevant table columns to make a new table
            combined_df = pd.concat([sub_df['opt_elem'], 
                                     sub_df['cenwave'],
                                     sub_df['segment'],
                                     sub_df['date-obs'].round(2),
                                     sub_df['relative_throughput'],
                                     sub_df['relative_throughput_err']],
                                     axis=1)

            tables.append(combined_df)
    
    df = pd.concat(tables)
            
    # save the csv -- maybe put in a date function for the name?
    return dcc.send_data_frame(df.to_csv, 'test_large_thruput.csv', index=False)

## ------- SAVING METHODS ------- ##
# save html files!!! save the solar flux
@callback(
    Output('solar-flux-html', 'data'),
    Input('save-solar-flux', 'n_clicks'),
    Input('solar-flux', 'figure')
)
def save_figure_as_html(n_clicks, figure):
    if n_clicks:
         # Convert the figure to HTML
        fig = go.Figure(figure)
        html_str = fig.to_html(full_html=False)

        # Trigger the download
        return dict(content=html_str, filename="test_solar_flux.html")

## ------- SAVING METHODS ------- ##
# save the time slope vs wavelength plots
@callback(
    Output('time-slope-html', 'data'),
    Input('save-time-slope', 'n_clicks'),
    Input('computed-results', 'data'),
    Input('dates', 'data'),
    Input('sizes', 'value')
)
def save_time_slope_as_html(n_clicks, data, dates, selected_size):

    if n_clicks is None:
        return dash.no_update
    if data is None:
        fig = go.Figure()
        return fig
    
    # !!! grab necessary data !!!
    # read in the fuv tds data as a dataframe
    df = pd.read_json(StringIO(data), orient='split')

    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)


    # look over breakpoints
    figs = []
    for i, bp in enumerate(monitor.breakpoints):
        # Establish plot
        fig = go.Figure()

        fig = monitor.time_slope_graph(df, selected_size, i, fig)

        figs.append(fig)
    

    html_str = ""
    for fig in figs:
        html_str+=fig.to_html(full_html=False)

    # Trigger the download
    return dict(content=html_str, filename="test_time_slope.html")

## ------- SAVING METHODS ------- ##
# save the relative sensitivity plots
@callback(
        Output('relative-html', 'data'),
        Input('save-relative-net', 'n_clicks'),
        Input('computed-results', 'data'),
        Input('dates', 'data'),
        Input('sizes', 'value')
)
def save_relative_as_html(n_clicks, data, dates, selected_size):

    if n_clicks is None:
        return dash.no_update
    if data is None:
        fig = go.Figure()
        return fig
    
    # !!! grab necessary data !!!
    # read in the fuv tds data as a dataframe
    df = pd.read_json(StringIO(data), orient='split')

    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)

    # to save as zip file
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # loop over all the things 
        for cenwave in df['cenwave'].unique():
            for segment in df['segment'][df['cenwave'] == cenwave].unique():
                sub_df = df[(df['cenwave'] == cenwave) & (df['segment'] == segment)]
                date = np.array(sub_df['date-obs']).flatten()
                grating = sub_df['opt_elem'].iloc[0]

                # scale to one
                scaled_df = monitor.scale_to_1(table=sub_df, size=selected_size) 

                net  = np.array([net for net in scaled_df[f'{selected_size}_scaled_net'].iloc[0]])
                net_err = np.array([net for net in scaled_df[f'{selected_size}_scaled_stdev'].iloc[0]])
                best_fit_model = np.array([net for net in scaled_df[f'{selected_size}_best_fit'].iloc[0]])

                binned_wl = np.array([wl for wl in sub_df[f'{selected_size}_binned_wl']])[0]
                wl_edges  = np.array([wl for wl in sub_df[f'{selected_size}_wl_edges']])[0]

                # TDS model
                tds = np.transpose([1.0 / monitor.tds_backout(
                    1.0, binned_wl, 
                    Time(mjd, format='decimalyear').mjd,
                    grating, 'ANY', segment, cenwave, TDSTAB
                ) for mjd in date])
                
                # loop over the binned wl
                figs = []
                for i, wl in enumerate(binned_wl):
                    fig = make_subplots(rows=2, cols=1, 
                                    shared_xaxes=True,
                                    vertical_spacing=0.05,
                                    subplot_titles=(f'{grating}/{cenwave}/{segment} - Relative net count rate {wl_edges[i]} - {wl_edges[i+1]} Å',
                                            ''))

                    fig, x = monitor.rel_sens_graph(date, net, net_err, i, wl_edges, best_fit_model, sub_df, tds, fig)

                    # update the x-axes
                    fig.update_xaxes(title_text='Date', range=(x[0]-0.1, x[-1]+0.1), row=2, col=1)
                    fig.update_yaxes(title_text='Relative net count rate', row=1, col=1),
                    fig.update_yaxes(title_text='Percent Difference', row=2, col=1)

                    figs.append(fig)
        
                # Trigger the download
                html_str = ""
                for fig in figs:
                    html_str+=fig.to_html(full_html=False)
                
                zip_file.writestr(f"test_relative_{cenwave}_{segment}.html", html_str)

    
    zip_buffer.seek(0)
    
    return dcc.send_bytes(zip_buffer.read(), "test_relative_sens.zip")
    

## ------- SAVING METHODS ------- ##
## ------- SAVING METHODS ------- ##


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

# Timebins
@callback(
    [Output('timebins', 'options'),
     Output('timebins', 'value')],
    [Input('computed-results', 'data'),
     Input('dates', 'data')])
def set_timebin_options(data, dates):
    if data is None:
        return "No results available."
    # establish the functions necessary for the analysis
    monitor = FUVTDSMonitor(dates)

    # populate the option dictionary for the different time bins
    options = []
    for i, bp in enumerate(monitor.breakpoints):

        if i == 0:
            options.append({'label': f't < {bp}', 'value': 0})
        elif i == len(monitor.breakpoints)-1:
            options.append({'label': f'{bp} < t', 'value': len(monitor.breakpoints)-1})
        else:
            options.append({'label': f'{monitor.breakpoints[i-1]} < t < {bp}', 'value': i})

    value = 0
    return(options, value)
## ------- BUTTONS ------- ##
## ------- BUTTONS ------- ##


if __name__ == '__main__':
    app.run(debug=True)