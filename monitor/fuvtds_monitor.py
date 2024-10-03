from dash import Dash, html, dcc, Input, Output, callback
from fuvtds_base_class import FUVTDSBase
from plotly.subplots import make_subplots
from astropy.time import Time
import numpy as np
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

TDSTAB = '/grp/hst/cdbs/lref/83j20454l_tds.fits'

df = FUVTDSBase(PIDs='fuvtds_analysis_list.dat')

#-------------------------------------------------------------------------------
#### App Layout ####
#-------------------------------------------------------------------------------

# Define the layout
app.layout = html.Div(children=[
    # Container for the overall layout
    html.Div(children=[
        # Container for the left side (Dropdowns)
        html.Div(children=[
            html.Label('Grating'),
            dcc.Dropdown(
                options=[{'label': opt, 'value': opt} for opt in df['opt_elem'].unique()],
                value='G130M',
                id='gratings'
            ),
            html.Br(),

            html.Label('Cenwave'),
            dcc.Dropdown(
                id='cenwaves'
            ),
            html.Br(),

            html.Label('Segment'),
            dcc.Dropdown(id='segments'),
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

    # New row for residuals and histogram
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(id='residuals')
        ], style={
            'width': '55%',
            'padding': '10px'
        }),
        html.Div(children=[
            dcc.Graph(id='histogram')
        ], style={
            'width': '45%',
            'padding': '10px'
        })
    ], style={'display': 'flex', 'width': '100%'})  # Flexbox for new row
])



## ------- PLOTS ------- ##
@callback(
       Output('relative-net', 'figure'),
       Input('gratings', 'value'),
       Input('cenwaves', 'value'),
       Input('segments', 'value'),
       Input('wavelength-bins', 'value'))
def update_graph(selected_grating, selected_cenwave, selected_segment, selected_wl_bin):

    # read in the pre-made csv file that has all the info we need
    df = pd.read_csv('inventory.csv')

    breakpoints = np.array([
        2019.0, 2020.6, 
        2022.0, 2023.2])
    HV_FUVA = np.array([
        2017.75,2020.75,
        2021.76, 2023.94])
    HV_FUVB = np.array([
        2017.75, 2020.75,
        2022.47, 2023.94])
    LPs = np.array([
        2017.75, 2021.76, 
        2022.75])

    # the selected wl bin becomes a string so we have to manually change it
    selected_wl_bin = float(selected_wl_bin)

    df = df[
        (df['opt_elem'] == selected_grating) & 
        (df['cenwave'] == selected_cenwave) & 
        (df['segment'] == selected_segment)]

    date = Time([time for time in df['date-obs']], format='fits').decimalyear

    curr_net = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['current_binned_net']])
    new_net  = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['new_binned_net']])

    binned_wl = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['binned_wl']])
    wl_edges = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['wl_edges']])

    # TDS model
    tds = np.transpose([1.0/tds_backout(
        1.0, binned_wl[0],
        Time(mjd, format='decimalyear').mjd,
        selected_grating, 'ANY',
        selected_segment, selected_cenwave, TDSTAB
    ) for mjd in date])

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{selected_grating}/{selected_cenwave}/{selected_segment} - Relative net count rate',
                                        ''))

    for i, wl in enumerate(binned_wl[0]):
        if wl == selected_wl_bin:
            curr_sens = go.Scatter(
                x=date,
                y=curr_net[:,i] / curr_net[:,i][0],
                mode='markers',
                name=f"Current TDS {wl_edges[0][i]} - {wl_edges[0][i+1]}",
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['HVLEVELA']).flatten(),
                        np.array(df['HVLEVELB']).flatten(),
                        np.array(df['date-obs']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'HV Level A/B: %{customdata[4]}/%{customdata[5]}<br>'+
                'date obs: %{customdata[6]}'
                "<extra></extra>",
                line_color='red'
            )

            new_sens = go.Scatter(
                x=date,
                y=new_net[:,i] / new_net[:,i][0],
                mode='markers',
                name=f"POSTGEO TDS {wl_edges[0][i]} - {wl_edges[0][i+1]}",
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['HVLEVELA']).flatten(),
                        np.array(df['HVLEVELB']).flatten(),
                        np.array(df['date-obs']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'HV Level A/B: %{customdata[4]}/%{customdata[5]}<br>'+
                'date obs: %{customdata[6]}'
                "<extra></extra>",
                line_color='green'
            )
            tds_model = go.Scatter(
                x = date,
                y = tds[i] / tds[i][0],
                name = 'TDSTAB',
                line = dict(color='orange', width=4, dash='dash')
            )

            residuals = go.Scatter(
                x=date,
                y= ( ((new_net[:,i] / new_net[:,i][0]) - (curr_net[:,i] / curr_net[:,i][0]))  / (curr_net[:,i] / curr_net[:,i][0])) ,
                mode='markers',
                name=f"Residuals {wl_edges[0][i]} - {wl_edges[0][i+1]}",
                customdata= np.stack(
                    (np.array(df['rootname']).flatten(),
                        np.array(df['life_adj']).flatten(),
                        np.array(df['proposid']).flatten(),
                        np.array(df['targname']).flatten(),            
                        np.array(df['HVLEVELA']).flatten(),
                        np.array(df['HVLEVELB']).flatten(),
                        np.array(df['date-obs']).flatten()),
                    axis=-1
                ),
                hovertemplate=
                'Rootname: %{customdata[0]}<br>'+
                'Life_adj: %{customdata[1]}<br>'+
                'Proposid: %{customdata[2]}<br>'+
                'Target: %{customdata[3]}<br>'+
                'HV Level A/B: %{customdata[4]}/%{customdata[5]}<br>'+
                'date obs: %{customdata[6]}'
                "<extra></extra>",
                line_color='blue'
            )

            # Fill between
            fig.add_trace(go.Scatter(
                x = [date[0]-0.1, date[-1]+0.1],
                y = [-0.05, -0.05],
                mode='lines',
                fill = 'tonexty',
                showlegend=False,
                line_color='teal'), row=2, col=1)
            fig.add_trace(go.Scatter(
                x = [date[0]-0.1, date[-1]+0.1],
                y = [0.05, 0.05],
                mode='lines',
                fill = 'tonexty',
                showlegend=False,
                line_color='teal'), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x = [date[0]-0.1, date[-1]+0.1],
                y = [-0.02, -0.02],
                mode='lines',
                fill = 'tozeroy',
                showlegend=False,
                line_color='purple'), row=2, col=1)
            fig.add_trace(go.Scatter(
                x = [date[0]-0.1, date[-1]+0.1],
                y = [0.02, 0.02],
                mode='lines',
                fill = 'tozeroy',
                showlegend=False,
                line_color='purple'), row=2, col=1)

            fig.add_trace(tds_model, row=1, col=1)
            fig.add_trace(curr_sens, row=1, col=1)
            fig.add_trace(new_sens, row=1, col=1)
            fig.add_trace(residuals, row=2, col=1)

            # add vertical lines here
            fig.add_traces(add_lines(breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint'))
            fig.add_traces(add_lines(breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint'), rows=2, cols=1)
            fig.add_traces(add_lines(LPs, dict(color='grey', width=2, dash='dot'), 'LP switch'))
            fig.add_traces(add_lines(LPs, dict(color='grey', width=2, dash='dot'), 'LP switch'), rows=2, cols=1)

            if selected_segment == 'FUVA':
                fig.add_traces(add_lines(HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA'))
                fig.add_traces(add_lines(HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA'), rows=2, cols=1)
            else:
                fig.add_traces(add_lines(HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB'))
                fig.add_traces(add_lines(HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB'), rows=2, cols=1)

    # add in the menu to the figure
    fig.update_xaxes(title_text='Date', range=(date[0]-0.1, date[-1]+0.1), row=2, col=1)
    fig.update_yaxes(title_text='Relative net count rate', row=1, col=1, range=(0, 1.25)),
    fig.update_yaxes(title_text='Residuals', range=(-0.25, 0.25), row=2, col=1)

    fig.update_layout(height=700)

    return (fig)

@callback(
       Output('residuals', 'figure'),
       Input('gratings', 'value'),
       Input('cenwaves', 'value'),
       Input('segments', 'value'))
def update_residuals(selected_grating, selected_cenwave, selected_segment):
    """
    """
    # read in the pre-made csv file that has all the info we need
    df = pd.read_csv('inventory.csv')

    df = df[
        (df['opt_elem'] == selected_grating) & 
        (df['cenwave'] == selected_cenwave) & 
        (df['segment'] == selected_segment)]

    curr_net = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['current_binned_net']])
    new_net  = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['new_binned_net']])

    binned_wl = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['binned_wl']])
    wl_edges = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['wl_edges']])

    fig = go.Figure()

    for i, wl in enumerate(binned_wl[0]):
        """
        """

        residual = go.Scatter(
            x = np.array([wl]),
            y = np.array([np.mean(( ((new_net[:,i] / new_net[:,i][0]) - (curr_net[:,i] / curr_net[:,i][0]))  / (curr_net[:,i] / curr_net[:,i][0])))]),
            mode='markers',
            name=f"Residuals {wl_edges[0][i]} - {wl_edges[0][i+1]}",
            customdata= np.stack(
                (np.array(df['rootname']).flatten(),
                    np.array(df['life_adj']).flatten(),
                    np.array(df['proposid']).flatten(),
                    np.array(df['targname']).flatten()),
                axis=-1
            ),
            hovertemplate=
            'Rootname: %{customdata[0]}<br>'+
            'Life_adj: %{customdata[1]}<br>'+
            'Proposid: %{customdata[2]}<br>'+
            'Target: %{customdata[3]}<br>'
            "<extra></extra>",
            showlegend=False
        )


        fig.add_trace(residual)

    fig.update_yaxes(title_text='Residuals', range=(-0.025, 0.025))
    fig.update_xaxes(title_text='Wavelength (Å)')
    fig.update_layout(title_text=f'{selected_grating}/{selected_cenwave}/{selected_segment}')
    return(fig)

## ------- PLOTS ------- ##
@callback(
       Output('histogram', 'figure'),
       Input('gratings', 'value'),
       Input('cenwaves', 'value'),
       Input('segments', 'value'))
def update_residual_histogram(selected_grating, selected_cenwave, selected_segment):
    # read in the pre-made csv file that has all the info we need
    df = pd.read_csv('inventory.csv')

    df = df[
        (df['opt_elem'] == selected_grating) & 
        (df['cenwave'] == selected_cenwave) & 
        (df['segment'] == selected_segment)]

    curr_net = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['current_binned_net']])
    new_net  = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['new_binned_net']])

    binned_wl = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['binned_wl']])

    residuals = []

    for i, _ in enumerate(binned_wl[0]):
        residuals.append(np.mean(( ((new_net[:,i] / new_net[:,i][0]) - (curr_net[:,i] / curr_net[:,i][0]))  / (curr_net[:,i] / curr_net[:,i][0]))))

    fig = go.Figure(data=[go.Histogram(x = residuals)])
    return(fig)



## ------- BUTTONS ------- ##
@callback(
    Output('cenwaves', 'options'),
    [Input('gratings', 'value')])
def set_cenwave_options(selected_grating):
    return [{'label': i, 'value': i} for i in df['cenwave'][df['opt_elem'] == selected_grating].unique()]

@callback(
    Output('cenwaves', 'value'),
    [Input('cenwaves', 'options')])
def set_cenwave_value(available_options):
    return available_options[0]['value']

@callback(
    Output('segments', 'options'),
    [Input('gratings', 'value'),
     Input('cenwaves', 'value')])
def set_segment_options(selected_grating, selected_cenwave):
    return [{'label': i, 'value': i} for i in df['segment'][(df['opt_elem'] == selected_grating) & (df['cenwave'] == selected_cenwave)].unique()]

@callback(
    Output('segments', 'value'),
    [Input('segments', 'options')])
def set_segment_value(available_options):
    return available_options[0]['value']

@callback(
    [Output('wavelength-bins', 'options'),
     Output('wavelength-bins', 'value')],
    [Input('gratings', 'value'),
     Input('cenwaves', 'value'),
     Input('segments', 'value')])
def set_wavelength_options(selected_grating, selected_cenwave, selected_segment):

    binned_wl = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['binned_wl'][(df['opt_elem'] == selected_grating) &
                                                                                                  (df['cenwave'] == selected_cenwave) &
                                                                                                  (df['segment'] == selected_segment)]])
    wl_edges  = np.array([ast.literal_eval(str(net).replace(' ', ',')) for net in df['wl_edges'][(df['opt_elem'] == selected_grating) &
                                                                                                  (df['cenwave'] == selected_cenwave) &
                                                                                                  (df['segment'] == selected_segment)]])

    for i, wl in enumerate(binned_wl):
        options = []
        for j, _ in enumerate(wl):
            options.append({'label': f'{wl_edges[i][j]} - {wl_edges[i][j+1]}',
                            'value': f'{binned_wl[i][j]}'})
        value = binned_wl[i][0]
        return(options, value)
## ------- BUTTONS ------- ##

if __name__ == '__main__':
    app.run(debug=True)