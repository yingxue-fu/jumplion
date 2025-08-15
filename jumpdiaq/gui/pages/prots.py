# builtin
import os
import ast
from collections import defaultdict
# external
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
import dash_bio
import plotly.graph_objects as go
import plotly.express as px
# local
from jumpdiaq.utils import extract_unique_parts
from jumpdiaq.gui.app import app, page_sidebar_style, logger
from jumpdiaq.gui.utils import get_valid_ratios, lfqbench_plot

load_figure_template(["vizro", "vizro_dark"])
colors = {'ALL': '#1F77B4',
          'HUMAN': '#32A481', 
          'YEAST': '#D1662B', 
          'ECOLI': '#7972B8',
          'MOUSE': '#32A481', 
          'SCHPO': '#D1662B', 
          }

sidebar = html.Div(
    [
        html.H3("Protein Quantification"),
        html.Hr(),
        dbc.Nav(
            [
                html.Label("Select a quantification table", style={"font-weight": "bold"}),
                dbc.Select(id='quant-table-selector', placeholder='Select a file'),
                html.Label("Input meta file path", style={"font-weight": "bold", 'marginTop': '10px'}),
                dbc.Input(id='meta-path', type='text', placeholder='Path of sample meta info', 
                          style={'width': '100%'}, persistence=True),
                dbc.Button('Load Quantification Table', 
                           id='load-prot-quant', n_clicks=0, color='info',
                           style={'margin-top': '10px', "textTransform": "none"}),
                html.Div(id='show-cutoff', style={'marginTop': '10px', 'color': 'green'}),
                html.Hr(),
                html.Label("Protein Intensity Distribution", style={"font-weight": "bold"}),
                dbc.Select(
                    id='species-input-violin',
                    options=[{'label': opt, 'value': opt} for opt in ['HUMAN', 'SCHPO', 'MOUSE', 'YEAST', 'ECOLI']],
                    placeholder="Select a species"
                ),
                html.Br(),
                html.Label("Protein Summarization", style={"font-weight": "bold", 'marginBottom': '5px'}),
                dbc.Switch(label='Show raw precursor quantity', id='show-raw', value=False),
                dbc.Switch(label='Show relative ratio of quantity', id='sub-row-means', value=False),
                dbc.Switch(label='Show summarized quantity/quality', id='show-summ', value=False),
                html.Br(),
                html.Label("Quantification Benchmark Plot", style={"font-weight": "bold"}),
                html.Label("Select groups for comparison", style={'marginTop': '5px'}),
                dbc.Select(id='group-A', placeholder="Group A"),
                dbc.Select(id='group-B', placeholder="Group B", style={'marginTop': '5px'}),
                html.Label("Input expected ratios:", style={'marginTop': '5px'}),
                dbc.Row([
                    dbc.Col(dbc.Input(id="ratio-1", type="number", value=1.0), width=4), 
                    dbc.Col(dbc.Input(id="ratio-2", type="number", value=2.0), width=4),
                    dbc.Col(dbc.Input(id="ratio-3", type="number", value=0.25), width=4),
                ]),
                html.Label("Select species"),
                dcc.Dropdown(
                    id='species-input-bench',
                    options=[{'label': opt, 'value': opt} for opt in ['HUMAN', 'SCHPO', 'MOUSE', 'YEAST', 'ECOLI']],
                    multi=True,
                    placeholder="Select one or more species"
                ),
                html.Label("Point Size:", style={'marginTop': '5px'}),
                dcc.Slider(
                    id='point-size',
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(1, 11)}
                ),
                html.Label("Point Opacity:", style={'marginTop': '5px'}),
                dcc.Slider(
                    id='point-opacity',
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    value=0.3,
                    marks={i/10: str(i/10) for i in range(1, 11)}
                ),
                html.Label("Y-axis Range:", style={'marginTop': '5px'}),
                dcc.Slider(
                    id='y-span',
                    min=1,
                    max=5,
                    step=1,
                    value=3,
                    marks={i: str(i) for i in range(1, 6)}
                ),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
    style=page_sidebar_style,
)

tab_label_style = {
    "textTransform": "none",  # Prevent all caps
    "textAlign": "center"
}

tab_bar_style = {
    "flexWrap": "nowrap",  # Prevent line wrapping
}

layout = html.Div(children=[
    dcc.Store(id='stored-prot-quant'),  # Store the protein quantification data
    dcc.Store(id='stored-prot-quality'),
    dcc.Store(id="stored-prot-prec-quant"),
    dcc.Store(id="stored-prot-prec-quant-norm"),
    dcc.Store(id="stored-prot-prec-quality"),
    dcc.Store(id="stored-valid-prots"),
    dbc.Row([
        dbc.Col(sidebar, width=4),
        dbc.Col([
            html.Br(),
            html.Div(children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("Protein quantification table")),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='prot-quant-table',
                            data=[],
                            columns=[],
                            row_selectable="single",
                            selected_rows=[],
                            style_table={
                                'overflowX': 'auto', 
                                'height': '400px', 
                                'overflowY': 'auto'},
                            style_cell={
                                'fontFamily': 'Arial, sans-serif',
                                'fontSize': '14px',
                                'textAlign': 'center', 
                                'padding': '5px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'maxWidth': '200px',
                                'color': 'black',  
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': 'white', #'lightgray',
                                'fontWeight': 'bold'},
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            page_size=10,
                            page_action="native",
                        ),
                    ])
                ], style={"width": "97%"})
            ], className="dbc dbc-row-selectable"),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Protein quantification plots")),
                    dbc.CardBody(
                        [
                            dbc.Tabs(
                                id="tabs",
                                active_tab="tab-1",
                                children=[
                                    dbc.Tab(label="Protein Intensity Distribution", 
                                            tab_id="tab-1", label_style=tab_label_style),
                                    dbc.Tab(label="Protein Quantity Summarization", 
                                            tab_id="tab-2", label_style=tab_label_style),
                                    dbc.Tab(label="Protein Quality Summarization", 
                                            tab_id="tab-3", label_style=tab_label_style),
                                    dbc.Tab(label="Protein Coverage Information", 
                                            tab_id="tab-4", label_style=tab_label_style),
                                    dbc.Tab(label="Quantification Benchmark Plot", 
                                            tab_id="tab-5", label_style=tab_label_style),
                                ],
                                style=tab_bar_style
                            ),
                            html.Div(
                                [   
                                    # Tab 1:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-violin-plot"),
                                        ],    
                                        id="tab-1-content",
                                    ),
                                    # Tab 2:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quant-plot"),
                                            dbc.Button("Check Precursors", id="check-prec-quantity", 
                                                       color="info",  
                                                       style={"font-weight": "bold",  
                                                              "marginTop": "5px",
                                                              "textTransform": "none"}),
                                        ],    
                                        id="tab-2-content",
                                    ),
                                    # Tab 3:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quality-plot"),
                                            dbc.Button("Check Precursors", id="check-prec-quality", 
                                                       color="info",  
                                                       style={"font-weight": "bold", 
                                                              "marginTop": "5px",
                                                              "textTransform": "none"}),
                                        ],    
                                        id="tab-3-content",
                                    ), 
                                    # Tab 4:
                                    html.Div(
                                        [
                                            html.Div(id='sequence-info'),
                                            html.Div(id='sequence-viewer-container'),
                                        ],    
                                        id="tab-4-content",
                                    ),  
                                    # Tab 5:
                                    html.Div(
                                        [
                                            html.Div([
                                                dcc.Graph(id='ratio-intensity-plot', style={'height': '500px'})
                                            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                            
                                            html.Div([
                                                dcc.Graph(id='ratio-violin-plot', style={'height': '500px'})
                                            ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        ], 
                                        id="tab-5-content",
                                    ),
                                ],
                                className="p-1"
                            )
                    ]
                )
            ],
            style={"marginTop": "5px", "width": "97%"}
            ),
        ], width=8)
    ])
])
   

@app.callback(
    Output('quant-table-selector', 'options'),
    Input('stored-outdir', 'data'),
)
def list_quant_files(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return []

    quant_tables = []
    quant_files = [f for f in os.listdir(dir_path) if f.startswith('Protein.Group_Quan_QT')]
    quant_tables.extend(quant_files)

    return [{'label': f, 'value': f} for f in quant_tables]
    

# Callback to read the selected CSV file
@app.callback(
    [Output('prot-quant-table', 'data'),
     Output('prot-quant-table', 'columns'),
     Output('stored-prot-quant', 'data'),
     Output('stored-prot-quality', 'data'),
     Output('show-cutoff', 'children'),
     Output('stored-meta', 'data')],
    Input('load-prot-quant', 'n_clicks'),
    State('quant-table-selector', 'value'),
    State('stored-outdir', 'data'),
    State('meta-path', 'value'),
    prevent_initial_call=True
)
def load_quant_table(n_clicks, selected_table, outdir_path, meta_path):
    
    if not selected_table or not outdir_path or not meta_path:
        return None, [], None, None, "", {}

    quant_csv_path = os.path.join(outdir_path, selected_table)
    quality_csv_path = os.path.join(outdir_path, "Protein.Group_Quan_Quality.csv")
    prot_pep_info_path = os.path.join(outdir_path, "prot_prec_pep_info.csv")
    
    try:
        quant_df = pd.read_csv(quant_csv_path)
        meta = pd.read_csv(meta_path)
        # Make the samples the same order as in the meta file
        sample_names = meta['Sample'].values.tolist()
        quant_df = quant_df.loc[:, quant_df.columns[0:3].values.tolist() + sample_names + [quant_df.columns[-1]]]
        # Simplify the sample name
        simple_sample_names = extract_unique_parts(sample_names)
        quant_df.columns = (
            quant_df.columns[0:3].values.tolist() + 
            simple_sample_names + 
            [quant_df.columns[-1]]
        )
        # Combine with protein coverage info
        prot_pep_df = pd.read_csv(prot_pep_info_path)
        quant_df = prot_pep_df.merge(quant_df, how='right')
        # Convert the string to an actual list of tuples
        quant_df['Peptide.Positions'] = quant_df['Peptide.Positions'].apply(ast.literal_eval)
        
        quant_df_to_show = quant_df.drop(columns=['Protein.Group', 'Protein.Description', 'Protein.Names', 'Sequence', 'Precursors', 'Peptides', 'Peptide.Positions'])
        value_cols = simple_sample_names + ['Average']
        quant_df_to_show[value_cols] = quant_df_to_show[value_cols].round(2)
        extracted_cutoffs = selected_table.split("QT")[1].split(".csv")[0].split("_quality")
        quality_tier, quality_cutoff = tuple(extracted_cutoffs)
        
        quality_df = pd.read_csv(quality_csv_path)
        quality_df = quality_df.loc[:, ['Protein.Group'] + meta['Sample'].values.tolist()]
        
        return (quant_df_to_show.to_dict('records'), 
                [{'name': col, 'id': col} for col in quant_df_to_show.columns], 
                quant_df.to_dict('records'), 
                quality_df.to_dict('records'),
                f"Using quality cutoff: {quality_cutoff}",
                meta.to_dict('records'))

    except Exception as e:
        logger.error(f"Error loading quantification table: {str(e)}")
        return None, [], None, None, "", {}


@app.callback(
    Output('species-input-bench', 'value'),
    Input('species-input-bench', 'value'),
    State('species-input-bench', 'value'),
    prevent_initial_call=True
)
def limit_selection(current_value, previous_value):
    if current_value is None:
        return []

    # Ensure current_value is always a list
    if not isinstance(current_value, list):
        current_value = [current_value]

    # Limit selections to at most 3
    if len(current_value) > 3:
        return previous_value  # Revert to previous value if limit exceeded
    return current_value


@app.callback(
    Output('prot-violin-plot', 'figure'),
    Input('stored-prot-quant', 'data'),
    Input('species-input-violin', 'value'),
)
def update_violin_plot(data, species_input):
    if data is None:
        return dash.no_update

    # Load data
    df = pd.DataFrame(data)
    species_column = "Protein.Names"
    value_columns = df.columns[12:-1]

    # Parse and clean user input
    species = [species_input] if species_input else []

    if not species:
        df_filtered = df.copy()
        df_filtered['Species'] = 'ALL'
    else:
        # Function to find first matching group in prot_names
        def define_species(prot_names):
            for s in species:
                if s in prot_names:
                    return s
            return None

        df['Species'] = df[species_column].apply(define_species)

        # Filter out rows with no matching group
        df_filtered = df[df['Species'].notna()]

        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')

    # Melt data for plotly
    df_melted = df_filtered.melt(id_vars=['Species'], value_vars=value_columns,
                                 var_name='Variable', value_name='Value')

    # Create figure
    fig = go.Figure()
    fig = px.violin(df_melted, 
                    x='Variable', 
                    y='Value', 
                    color='Species', 
                    color_discrete_map=colors,
                    box=True, 
                    points='outliers')

    fig.update_layout(
        title='', 
        xaxis_title="Runs",
        yaxis_title="Log2(Intensity)",
        violingap=0,
        template="vizro",
        height=700,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    return fig


@app.callback(
    Output("tab-1-content", "style"),
    Output("tab-2-content", "style"),
    Output("tab-3-content", "style"),
    Output("tab-4-content", "style"),
    Output("tab-5-content", "style"),
    Input("tabs", "active_tab")
)
def toggle_tab(tab):
    styles = {"display": "none"}
    if tab == "tab-1":
        return {}, styles, styles, styles, styles
    elif tab == "tab-2":
        return styles, {}, styles, styles, styles
    elif tab == "tab-3":
        return styles, styles, {}, styles, styles
    elif tab == "tab-4":
        return styles, styles, styles, {}, styles
    else:
        return styles, styles, styles, styles, {}

@app.callback(
    Output('group-A', 'options'),
    Output('group-B', 'options'),
    Input('stored-meta', 'data'),
    prevent_initial_call=True
)
def generate_group_options(meta):
    if meta is None:
        return [], []

    meta = pd.DataFrame(meta)
    if 'Group' not in meta.columns:
        return [], []

    groups = sorted(meta['Group'].dropna().unique().tolist())
    group_options = [{'label': group, 'value': group} for group in groups]

    return group_options, group_options

# Callback to update plots
@app.callback(
    [Output('ratio-intensity-plot', 'figure'),
     Output('ratio-violin-plot', 'figure'),
     Output('stored-valid-prots', 'data')],
    [Input('species-input-bench', 'value'),
     Input('ratio-1', 'value'),
     Input('ratio-2', 'value'),
     Input('ratio-3', 'value'),
     Input('group-A', 'value'),
     Input('group-B', 'value'),
     Input('point-size', 'value'),
     Input('point-opacity', 'value'),
     Input('y-span', 'value'),
     Input('stored-prot-quant', 'data'),
     State('stored-meta', 'data')],
    prevent_initial_call=True
)
def update_benchmark_plot(species_input, ratio_1, ratio_2, ratio_3, group_A, group_B, point_size, point_opacity, y_span, data, meta):
    if data is None:
        return dash.no_update, dash.no_update, {}
    
    # Load data
    df = pd.DataFrame(data)
    meta = pd.DataFrame(meta)
    
    species_column = "Protein.Names"
    tag_column = "Protein.Group"
    value_columns = extract_unique_parts(meta['Sample'].to_list())
    
    # Parse and clean user input
    species = [s.strip() for s in species_input] if species_input else []
    exp_ratios = [ratio_1, ratio_2, ratio_3]
    
    if not species:
        df_filtered = df.copy()
        df_filtered['Species'] = 'ALL'
    else:
        # Function to find first matching group in prot_names
        def define_species(prot_name):
            if pd.isna(prot_name):
                return None
            for s in species:
                if s in prot_name:
                    return s
            return None

        df['Species'] = df[species_column].apply(define_species)
        # Filter out rows with no matching group
        df_filtered = df[df['Species'].notna()]

        if df_filtered.empty:
            empty_fig = px.scatter(title='No matching species found in your input.')
            return empty_fig, empty_fig, {}
    
    protein_tags = df_filtered[tag_column].values.tolist()
    protein_species = df_filtered['Species'].values.tolist()
    species_list = df_filtered['Species'].unique().tolist()
    
    cols_A = meta.index[meta['Group'] == group_A]
    cols_B = meta.index[meta['Group'] == group_B]  
    
    # Create LFQBench plots
    mat = df_filtered[value_columns]
    valid_prots = get_valid_ratios(mat, protein_tags, protein_species, species_list, 
                                 sample_groups=[group_A, group_B], 
                                 cols_A=cols_A, cols_B=cols_B)
    scatter_fig, violin_fig = lfqbench_plot(
        valid_prots, species_list, exp_ratios, 
        colors=colors, span=y_span, point_cex=point_size, 
        sample_groups=[group_A, group_B], title="",
        opacity=point_opacity
    )
    
    scatter_fig.update_layout(clickmode='event+select')
    
    valid_prots_all = [item for sublist in valid_prots.values() for item in sublist]
    return scatter_fig, violin_fig, {tag: (x, y) for tag, x, y in valid_prots_all}


# Separate callback to update highlight on scatter plot
@app.callback(
    Output('ratio-intensity-plot', 'figure', allow_duplicate=True),
    Input('selected-prot', 'data'),
    State('ratio-intensity-plot', 'figure'),
    State('stored-valid-prots', 'data'),
    State('point-size', 'value'),
    prevent_initial_call=True
)
def update_highlight(selected_prot, current_fig, valid_prots, point_size):
    if not selected_prot or not current_fig or not valid_prots:
        return dash.no_update
        
    # Create new figure based on current figure
    new_fig = go.Figure(current_fig)
    
    # Remove any existing highlight traces (traces with name 'highlight')
    new_fig.data = [trace for trace in new_fig.data if trace.name != 'highlight']
    
    # Add new highlight if protein exists in valid_prots
    if selected_prot in valid_prots:
        highlight_dot = valid_prots[selected_prot]
        new_fig.add_trace(
            go.Scatter(
                x=[highlight_dot[0]],
                y=[highlight_dot[1]],
                mode='markers',
                marker=dict(color='red', size=point_size*15),
                showlegend=False,
                hoverinfo='skip',
                name='highlight'  # Name to identify highlight traces
            )
        )
    
    return new_fig


@app.callback(
    Output('selected-prot', 'data'),
    Input('ratio-intensity-plot', 'clickData'),
    Input('prot-quant-table', 'selected_rows'),
    State('stored-prot-quant', 'data'),
)
def update_selected(clickData, selected_rows, table_data):
    # Check if callback was triggered
    if not ctx.triggered:
        return dash.no_update
    
    triggered_id = ctx.triggered_id
    
    try:
        if triggered_id == 'ratio-intensity-plot' and clickData:
            selected_prot = clickData['points'][0].get('text')
            return selected_prot
        elif triggered_id == 'prot-quant-table' and selected_rows and table_data:
            df = pd.DataFrame(table_data)
            if len(selected_rows) > 0 and selected_rows[0] < len(df):
                selected_prot = df.iloc[selected_rows[0]].get("Protein.Group", "")
                return selected_prot
    except Exception as e:
        print(f"Error in update_selected: {e}")
        return dash.no_update
    
    return dash.no_update


@app.callback(
    [Output('stored-prot-prec-quant', 'data'),
     Output('stored-prot-prec-quant-norm', 'data'),
     Output('stored-prot-prec-quality', 'data'),
     Output('stored-precursor-options', 'data')],
    Input('selected-prot', 'data'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
)
def read_precs_data(selected_prot, outdir_path, meta):
    
    if not selected_prot or outdir_path is None:
        return None, None, None, []

    meta = pd.DataFrame(meta)

    # Assuming "file_name" column contains the name of the corresponding HDF file
    prec_quant_file = "Precursor.Id_quant_summ_table.parquet"
    prec_quant_norm_file = "Precursor.Id_quant_summ_table_Norm.parquet"
    prec_quality_file = "Precursor.Id_quality_summ_table.parquet"
    prec_quant_path = os.path.join(outdir_path, prec_quant_file)
    prec_quant_norm_path = os.path.join(outdir_path, prec_quant_norm_file)
    prec_quality_path = os.path.join(outdir_path, prec_quality_file)
    
    try:
        # read quant data
        quant_df = pd.read_parquet(prec_quant_path, filters=[('Protein.Group', '==', selected_prot)])
        precursors = np.array(quant_df['Precursor.Id_MS_type'].tolist())
        precursors = np.unique(precursors)
        quant_df.set_index('Precursor.Id_MS_type', drop=True, inplace=True)
        quant_df = quant_df.loc[:, meta['Sample'].values.tolist()]
        quant_df.columns = extract_unique_parts(quant_df.columns.values.tolist())
        quant_df_long = quant_df.reset_index().melt(id_vars='Precursor.Id_MS_type', var_name='X', value_name='Y')
        
        # read quant norm data
        quant_df_norm = pd.read_parquet(prec_quant_norm_path, filters=[('Protein.Group', '==', selected_prot)])
        quant_df_norm.set_index('Precursor.Id_MS_type', drop=True, inplace=True)
        quant_df_norm = quant_df_norm.loc[:, meta['Sample'].values.tolist()]
        quant_df_norm.columns = extract_unique_parts(quant_df_norm.columns.values.tolist())
        quant_df_norm_long = quant_df_norm.reset_index().melt(id_vars='Precursor.Id_MS_type', var_name='X', value_name='Y')
        
        # read quality data
        quality_df = pd.read_parquet(prec_quality_path, filters=[('Protein.Group', '==', selected_prot)])
        quality_df.set_index('Precursor.Id_MS_type', drop=True, inplace=True)
        quality_df = quality_df.loc[:, meta['Sample'].values.tolist()]
        quality_df.columns = extract_unique_parts(quality_df.columns.values.tolist())
        quality_df_long = quality_df.reset_index().melt(id_vars='Precursor.Id_MS_type', var_name='X', value_name='Y')
        
        return (quant_df_long.to_dict("records"), 
                quant_df_norm_long.to_dict("records"), 
                quality_df_long.to_dict("records"), 
                precursors)
    except Exception as e:
        logger.error(f"Error loading parquet file: {str(e)}")
        return None, None, None, []
    

# Callback to generate figure from selected row
@app.callback(
    Output('prot-prec-quant-plot', 'figure'),
    Output('prot-prec-quality-plot', 'figure'),
    [Input('selected-prot', 'data'),
     Input('show-raw', 'value'), 
     Input('sub-row-means', 'value'),
     Input('show-summ', 'value'),
     Input('stored-prot-prec-quant', 'data'),
     Input('stored-prot-prec-quant-norm', 'data'),
     Input('stored-prot-prec-quality', 'data')],  
    [State('stored-prot-quant', 'data'),
     State('stored-prot-quality', 'data')],
    prevent_initial_call=True  # Add this to prevent initial callback
)
def update_summ_plot(selected_prot, 
                     show_raw_intensity, 
                     sub_row_means, 
                     show_summ_value, 
                     prec_quant_data, 
                     prec_quant_norm_data, 
                     prec_quality_data,
                     quant_data, 
                     quality_data):
    if not ctx.triggered or not selected_prot or not quant_data or not prec_quant_data:
        return (go.Figure(layout=dict(title="Select a protein to visualize protein quantification data")),
                go.Figure(layout=dict(title="Select a protein to visualize protein quality data")))
    
    try:
        # Retrieve selected row data
        quant_df = pd.DataFrame(quant_data)
        selected_row_data = quant_df[quant_df["Protein.Group"] == selected_prot].iloc[0,:]
        
        #
        quality_df = pd.DataFrame(quality_data)
        quality_df.set_index('Protein.Group', drop=True, inplace=True)
        quality_df.columns = extract_unique_parts(quality_df.columns.values.tolist())
        
        # Make sure we're getting numerical values for the plot
        prot_quant_values = pd.to_numeric(selected_row_data.iloc[12:-1], errors='coerce').values
        prot_quality_values = pd.to_numeric(quality_df.loc[selected_prot,:], errors='coerce').values
        
        # Create figure
        quant_fig = go.Figure()
        quality_fig = go.Figure()
        
        if show_raw_intensity:
            prec_quant_df = pd.DataFrame(prec_quant_data)
        else:
            prec_quant_df = pd.DataFrame(prec_quant_norm_data)
        prec_quality_df = pd.DataFrame(prec_quality_data)
        
        # Plot each tract as a connected scatter plot
        unique_precursors = pd.concat([
            prec_quant_df["Precursor.Id_MS_type"], 
            prec_quality_df["Precursor.Id_MS_type"]
        ]).unique()
        num_traces = len(unique_precursors)
        if num_traces <= len(px.colors.qualitative.Plotly):
            sampled_colors = px.colors.qualitative.Plotly[:num_traces]
        else:
            default_colorscale = [[i/(len(px.colors.qualitative.Plotly)-1), color] 
                                  for i, color in enumerate(px.colors.qualitative.Plotly)]
            sampled_colors = px.colors.sample_colorscale(default_colorscale, num_traces)
        trace_colors = dict(zip(unique_precursors, sampled_colors))
        for tract_id, group in prec_quant_df.groupby("Precursor.Id_MS_type"):
            if sub_row_means:
                quant_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"]-np.nanmean(group["Y"].to_numpy()), 
                    mode="lines+markers",
                    name="Precursors" if show_summ_value else tract_id,
                    line=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                    marker=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                    showlegend=not show_summ_value,
                    opacity=0.7
                ))
            else:
                quant_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"], 
                    mode="lines+markers",
                    name="Precursors" if show_summ_value else tract_id,
                    line=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                    marker=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                    showlegend=not show_summ_value
                ))
        
        for tract_id, group in prec_quality_df.groupby("Precursor.Id_MS_type"):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                marker=dict(color='gray' if show_summ_value else trace_colors[tract_id]),
                showlegend=not show_summ_value,
                opacity=0.7 if show_summ_value else 1.0
            ))
        
        if show_summ_value:
            if sub_row_means:
                summ_values = prot_quant_values - np.nanmean(prot_quant_values)
            else:
                summ_values = prot_quant_values
            summarization_x = quant_df.columns[12:-1].tolist()
            quant_fig.add_trace(go.Scatter(
                x=summarization_x, 
                y=summ_values, 
                mode="lines+markers",
                name="Summarization",
                line=dict(color='blue', width=2),
                marker=dict(color='blue', size=8)
            ))
            
            summarization_x = quality_df.columns.tolist()
            quality_fig.add_trace(go.Scatter(
                x=summarization_x, 
                y=prot_quality_values, 
                mode="lines+markers",
                name="Summarization",
                line=dict(color='blue', width=2),
                marker=dict(color='blue', size=8)
            ))
        
        # Layout settings
        quant_fig.update_layout(
            title=selected_prot,
            xaxis_title="Runs",
            yaxis_title="Log2(Ratio)" if sub_row_means else "Log2(Intensity)",
            template="vizro",
            height=500,
            xaxis=dict(domain=[0, 0.7]),  # Plot uses only 80% of width
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.72,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                orientation="v",
                traceorder="normal"
            ),
        )
        
        quality_fig.update_layout(
            title=selected_prot,
            xaxis_title="Runs",
            yaxis_title="Quality",
            template="vizro",
            height=500,
            xaxis=dict(domain=[0, 0.7]),  # Plot uses only 80% of width
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.72,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                orientation="v",
                traceorder="normal", 
            ),
        )
        
        # 
        quant_fig.update_traces(hoverinfo='name')
        quality_fig.update_traces(hoverinfo='name')
        
        return quant_fig, quality_fig
    except Exception as e:
        # Return an error message in the figure
        logger.error(f"Error updating plot: {str(e)}")
        return (go.Figure(layout=dict(title=f"Error updating plot: {str(e)}")),
                go.Figure(layout=dict(title=f"Error updating plot: {str(e)}")))


@app.callback(
    Output('sequence-viewer-container', 'children'),
    Output('sequence-info', 'children'),
    Input('selected-prot', 'data'),
    State('stored-prot-quant', 'data')
)
def update_sequence_viewer(selected_prot, data):
    if not ctx.triggered or not selected_prot or not data:
        return (html.Div("Select a protein to show its coverage information", 
                        style={'fontSize': '20px', 
                               'marginTop': '15px',
                               'marginLeft': '20px'}), 
                None)
    
    # Retrieve selected row data
    df = pd.DataFrame(data)
    row = df[df["Protein.Group"] == selected_prot].iloc[0,:]
    
    selected_protein = row["Accession.Number"]
    sequence = row['Sequence']
    peptides = row['Peptides'].split(',')
    peptides_pos = row['Peptide.Positions']
    coverage_percent = row['Coverage.Percentage']
    
    # Track coverage count per position
    position_coverage = defaultdict(int)
    
    # Step 1: Count how many times each position is covered
    for peptide_pos in peptides_pos:
        start = peptide_pos[0] - 1
        end = peptide_pos[1]
        for i in range(start, end):
            position_coverage[i] += 1
    
    # Step 2: Build non-overlapping segments
    sorted_positions = sorted(position_coverage.keys())
    segments = []
    if sorted_positions:
        current_start = sorted_positions[0]
        current_coverage = position_coverage[current_start]
    
        for i in range(1, len(sorted_positions)):
            pos = sorted_positions[i]
            if pos != sorted_positions[i-1] + 1 or position_coverage[pos] != current_coverage:
                segments.append((current_start, sorted_positions[i-1] + 1, current_coverage))
                current_start = pos
                current_coverage = position_coverage[pos]
        segments.append((current_start, sorted_positions[-1] + 1, current_coverage))
    
    # Step 3: Assign color based on coverage
    def get_color(coverage_count):
        if coverage_count == 1:
            return '#B6E880'  # Original color
        else:
            return '#FECB52'  # Overlapping region color (adjust as needed)
    
    # Step 4: Generate coverage_data without overlaps
    coverage_data = []
    for start, end, count in segments:
        coverage_data.append({
            'start': start,
            'end': end,
            'tooltip': f"Coverage: {count} peptide(s)",
            'bgcolor': get_color(count),
            'color': 'black',
            'underscore': False
        })

    
    # Create sequence viewer with correctly formatted coverage parameter
    sequence_viewer = dash_bio.SequenceViewer(
        sequence=sequence,
        title=f"{row['Genes']}",
        sequenceMaxHeight='300px',
        showLineNumbers=True,
        charsPerLine=100,
        toolbar=True,
        wrapAminoAcids=True,
        coverage=coverage_data,  # Using the properly formatted coverage data
    )
    
    # Create info section
    info_section = html.Div([
        html.P(f"Accession Number: {selected_protein}"),
        html.P(f"Protein Description: {row['Protein.Description']}"),
        html.P(f"Number of Peptides: {len(peptides)}"),
        html.P(f"Peptides: {row['Peptides']}"),
        html.P(f"Coverage: {coverage_percent:.1f}%"),
    ], style={'marginBottom': '20px'})
    
    return sequence_viewer, info_section