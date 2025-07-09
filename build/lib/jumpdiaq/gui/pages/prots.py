# builtin
import os
import ast
# external
import numpy as np
import pandas as pd
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, callback_context, no_update
import plotly.graph_objects as go
import plotly.express as px
import dash_bio as dashbio
from collections import defaultdict
# local
from app import app, page_sidebar_style, logger
from utils import extract_unique_parts, lfqbench_plot

load_figure_template(["vizro", "vizro_dark"])


sidebar = html.Div(
    [
        html.H3("Protein Quantification"),
        html.Hr(),
        dbc.Nav(
            [
                html.P("Select a quantification table"),
                dcc.Dropdown(id='quant-table-selector', placeholder='Select a file', multi=False),
                html.Button('Load Quantification Table', id='load-prot-quant', n_clicks=0, style={'margin-top': '10px'}),
                html.Div(id='show-cutoff', style={'marginTop': '10px', 'color': 'green'}),
                html.Hr(),
                html.Label("Overall Intensity", style={"font-weight": "bold"}),
                html.P("Select a species"),
                dcc.Dropdown(
                    id='species-input-violin',
                    options=[{'label': opt, 'value': opt} for opt in ['HUMAN', 'SCHPO', 'MOUSE', 'YEAST', 'ECOLI']],
                    multi=False,
                    placeholder='Species',
                    searchable=True,
                ),
                html.Hr(),
                html.Label("Benchmark Plot", style={"font-weight": "bold"}),
                html.P("Select one or more species"),
                dcc.Dropdown(
                    id='species-input-bench',
                    options=[{'label': opt, 'value': opt} for opt in ['HUMAN', 'SCHPO', 'MOUSE', 'YEAST', 'ECOLI']],
                    multi=True,
                    placeholder='Species',
                    searchable=True,
                ),
                html.P("Group A"),
                dcc.Dropdown(id='group-A', multi=False, 
                             placeholder="Group A", className = "dbc"),
                html.P("Group B"),
                dcc.Dropdown(id='group-B', multi=False, 
                             placeholder="Group B", className = "dbc"),
                html.Br(),
                html.Label("Expected Ratios:"),
                dbc.Row([
                    dbc.Col(dcc.Input(id="ratio-1", type="number", value=1.0, style={"width": "70px"}), width=3), 
                    dbc.Col(dcc.Input(id="ratio-2", type="number", value=1.0, style={"width": "70px"}), width=3),
                    dbc.Col(dcc.Input(id="ratio-3", type="number", value=1.0, style={"width": "70px"}), width=3),
                ], className="mb-2"),
                #dcc.Input(id='ratios-input', type='text', placeholder='Enter expected ratios'),
                html.Div([
                    html.Label("Point Size:"),
                    dcc.Slider(
                        id='point-size',
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(1, 11)}
                    ),
                ], style={'width': '100%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Y-axis Range:"),
                    dcc.Slider(
                        id='y-span',
                        min=1,
                        max=5,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 6)}
                    ),
                ], style={'width': '100%', 'display': 'inline-block'}),
                html.Hr(),
                html.Label("Protein Summarization", style={"font-weight": "bold", 'marginBottom': '10px'}),
                dbc.Switch(label='Substract row means', id='sub-row-means', 
                           value=False, className="d-inline-block"),
                dbc.Switch(label='Show Summarization Value', id='show-summ', 
                           value=False, className="d-inline-block"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
    style=page_sidebar_style,
)


layout = html.Div(children=[
    dcc.Store(id='stored-prot-quant'),  # Store the protein quantification data
    dcc.Store(id='stored-prot-quality'),
    dcc.Store(id="stored-prot-prec-quant"),
    dcc.Store(id="stored-prot-prec-quality"),
    
    dbc.Row([
        dbc.Col(sidebar, width=3, md={"size": 3, "offset": 1}),
        dbc.Col([
            html.Br(),
            html.Div(children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("Protein quantification table")),
                    dbc.CardBody([
                        dash_table.DataTable(  # Empty DataTable initially
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
                                'textAlign': 'center', 
                                'padding': '5px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'maxWidth': '200px',
                                'color': 'black',  
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': 'lightgray', 
                                'fontWeight': 'bold'},
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            page_size=10,
                            page_action="native",
                            
                        ),
                    ])
                ], className="mx-auto")
            ], className="dbc dbc-row-selectable"),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Protein quantification plots")),
                    dbc.CardBody(
                        [
                            dcc.Tabs(
                                id="tabs",
                                value="tab-1",
                                children=[
                                    dcc.Tab(label="Overall Intensity", value="tab-1"),
                                    dcc.Tab(label="Benchmark Plot", value="tab-2"),
                                    dcc.Tab(label="Protein Summarization", value="tab-3"),
                                    dcc.Tab(label="Protein Quan-Quality", value="tab-4"),
                                    dcc.Tab(label="Protein Coverage ", value="tab-5"),
                                ]
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
                                    html.Div([
                                        html.Div([
                                            dcc.Graph(id='ma-plot', style={'height': '500px'})
                                        ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        
                                        html.Div([
                                            dcc.Graph(id='box-hist-plot', style={'height': '500px'})
                                        ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        
                                        # html.Div([
                                        #     dcc.Graph(id='sd-plot', style={'height': '500px'})
                                        # ], style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'})
                                    ], id="tab-2-content",),
                                    # Tab 3:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quant-plot"),
                                        ],    
                                        id="tab-3-content",
                                    ),
                                    # Tab 4:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quality-plot"),
                                        ],    
                                        id="tab-4-content",
                                    ), 
                                    # Tab 5:
                                    html.Div(
                                        [
                                            html.Div(id='sequence-info'),
                                            html.Div(id='sequence-viewer-container'),
                                        ],    
                                        id="tab-5-content",
                                    ),  
                                ],
                                className="p-4"
                            )
                    ]
                )
            ],
            className="mt-4 shadow"
            ),
        ], width=8, className="mx-auto")
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

    # Find .mzML files
    quant_files = [f for f in os.listdir(dir_path) if f.startswith('Protein.Group_Quan_QT')]
    quant_tables.extend(quant_files)

    return [{'label': f, 'value': f} for f in quant_tables]


# Callback to read and store CSV file in DataFrame
@app.callback(
    Output('stored-meta', 'data'),
    Input('stored-meta-path', 'data')
)
def read_meta(meta_path):
    if meta_path is None:
        return {}

    # Decode the uploaded CSV file
    try:
        # Read the CSV into a pandas dataframe
        df = pd.read_csv(meta_path)
        
        # Return the DataFrame as a dictionary (this is how we store it)
        return df.to_dict('records')  
    except Exception:
        return {}
    

# Callback to read the selected CSV file
@app.callback(
    [Output('prot-quant-table', 'data'),
     Output('prot-quant-table', 'columns'),
     Output('stored-prot-quant', 'data'),
     Output('stored-prot-quality', 'data'),
     Output('show-cutoff', 'children')],
    Input('load-prot-quant', 'n_clicks'),
    State('quant-table-selector', 'value'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
    prevent_initial_call=True
)
def load_quant_table(n_clicks, selected_table, outdir_path, meta):
    
    if not selected_table or not outdir_path:
        return None, [], None, None, ""
    # outdir_path=r"C:\Users\yfu\Downloads\miss_dia_test_data\trixi_data\jumpdiaq_test2"
    quant_csv_path = os.path.join(outdir_path, selected_table)
    quality_csv_path = os.path.join(outdir_path, "Protein.Group_Quan_Quality.csv")

    try:
        quant_df = pd.read_csv(quant_csv_path)
        meta = pd.DataFrame(meta)
        quant_df = quant_df.loc[:, quant_df.columns[0:12].values.tolist() + meta['Sample'].values.tolist() + [quant_df.columns[-1]]]
        quant_df.columns = (
            quant_df.columns[0:12].values.tolist() + 
            extract_unique_parts(quant_df.columns[12:-1].values.tolist()) + 
            [quant_df.columns[-1]]
        )
        # Convert the string to an actual list of tuples
        quant_df['Peptide.Positions'] = quant_df['Peptide.Positions'].apply(ast.literal_eval)
        
        quant_df_to_show = quant_df.drop(columns=['Protein.Group', 'Protein.Description', 'Protein.Names', 'Sequence', 'Precursors', 'Peptides', 'Peptide.Positions'])
        cutoff = selected_table.split("QT")[1].split(".csv")[0]
        
        quality_df = pd.read_csv(quality_csv_path)
        quality_df = quality_df.drop(columns=['Accession.Number', 'Protein.Description', 'Sequence', 'Precursors', 'n.Precursors', 'Peptides', 'n.Peptides', 'Peptide.Positions', 'Coverage.Percentage', 'Protein.Names', 'Genes', 'Average'])
        
        return (quant_df_to_show.to_dict('records'), 
                [{'name': col, 'id': col} for col in quant_df_to_show.columns], 
                quant_df.to_dict('records'), 
                quality_df.to_dict('records'),
                f"Using quality cutoff: {cutoff}")

    except Exception as e:
        logger.error(f"Error loading quantification table: {str(e)}")
        return None, [], None, None, ""


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

    # Limit selections to at most 2
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
        return no_update

    # Load data
    df = pd.DataFrame(data)

    tag_column = "Protein.Names"
    value_columns = df.columns[12:-1]

    # Parse and clean user input
    species = [species_input] if species_input else []

    if not species:
        df_filtered = df.copy()
        df_filtered['Species'] = 'All'
    else:
        # Function to find first matching group in prot_tags
        def define_species(prot_tags):
            for s in species:
                if s in prot_tags:
                    return s
            return None

        df['Species'] = df[tag_column].apply(define_species)

        # Filter out rows with no matching group
        df_filtered = df[df['Species'].notna()]

        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')

    # Melt data for plotly
    df_melted = df_filtered.melt(id_vars=['Species'], value_vars=value_columns,
                                 var_name='Variable', value_name='Value')

    # Create figure
    fig = go.Figure()
    #species_values = df_melted['Species'].unique()
    #colors = ['#636EFA', '#EF553B']
    # if len(species_values) == 2:
    #     for species, side, color in zip(species_values, ['negative', 'positive'], colors):
    #         fig.add_trace(go.Violin(
    #             x=df_melted.loc[df_melted['Species'] == species, 'Variable'],
    #             y=df_melted.loc[df_melted['Species'] == species, 'Value'],
    #             name=str(species),
    #             side=side,
    #             box_visible=True,
    #             points='outliers',
    #             line_color=color,
    #             #fillcolor=color,
    #             meanline_visible=True
    #         ))
    # else:
    #     # fallback to normal combined violin plot
    fig = px.violin(df_melted, x='Variable', y='Value', color='Species',
                    box=True, points='outliers')

    fig.update_layout(
        title='', 
        xaxis_title="Runs",
        yaxis_title="log2(intensity)",
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
    Input("tabs", "value")
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
    [Output('ma-plot', 'figure'),
     Output('box-hist-plot', 'figure')],
    [Input('stored-prot-quant', 'data'),
     Input('species-input-bench', 'value'),
     Input('ratio-1', 'value'),
     Input('ratio-2', 'value'),
     Input('ratio-3', 'value'),
     Input('group-A', 'value'),
     Input('group-B', 'value'),
     Input('point-size', 'value'),
     Input('y-span', 'value'),
     #Input('sd-ylim', 'value'),
     State('stored-meta', 'data')]
)
def update_benchmark_plot(data, species_input, ratio_1, ratio_2, ratio_3, group_A, group_B, point_size, y_span, meta):
    if data is None:
        return no_update
    
    # Load data
    # df = pd.read_csv(r"C:\Users\yfu\Downloads\miss_dia_test_data\trixi_data\jumpdiaq_test2\Protein.Group_Quan_maxLFQ_Norm.csv")
    # meta = pd.read_csv(r"C:\Users\yfu\Downloads\miss_dia_test_data\trixi_data\sample_meta.csv")
    df = pd.DataFrame(data)
    
    tag_column = "Protein.Names"
    value_columns = df.columns[12:-1].values
    
    # Parse and clean user input
    species = [s.strip() for s in species_input] if species_input else []
    ratios = [ratio_1, ratio_2, ratio_3]
    
    if not species:
        df_filtered = df.copy()
        df_filtered['Species'] = 'All'
    else:
        # Function to find first matching group in prot_tags
        def define_species(prot_tags):
            for s in species:
                if s in prot_tags:
                    return s
            return None

        df['Species'] = df[tag_column].apply(define_species)

        # Filter out rows with no matching group
        df_filtered = df[df['Species'].notna()]

        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')
    
    protein_tags = df_filtered['Species'].values.tolist()
    species_list = df_filtered['Species'].unique().tolist()
    num_species = len(species_list)
    
    meta = pd.DataFrame(meta)
    # Set up parameters
    if num_species == 1:
        colors = ['#D3D3D3']
    elif num_species == 2:
        colors = ['#53868B', '#7B3F00']
    elif num_species == 3:
        colors = ['#32A481', '#D1662B', '#7972B8']

    cols_A = meta.index[meta['Group'] == group_A]
    cols_B = meta.index[meta['Group'] == group_B]  
    
    # Create LFQBench plots
    mat = df_filtered[value_columns]
    scatter_fig, box_hist_fig = lfqbench_plot(
        mat, protein_tags, species_list, ratios, 
        colors=colors, cols_A=cols_A, cols_B=cols_B,
        span=y_span, point_cex=point_size, #SDylim=[0,sd_ylim],
        sample_groups=[group_A, group_B],
    )
    
    return scatter_fig, box_hist_fig#, sd_fig


@app.callback(
    [Output('stored-prot-prec-quant', 'data'),
     Output('stored-prot-prec-quality', 'data'),
     Output('selected-prot', 'data'),
     #Output('prot-precursor-dropdown', 'options'),
     Output('stored-precursor-options', 'data')],
    Input('prot-quant-table', 'selected_rows'),
    State('stored-prot-quant', 'data'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
    #prevent_initial_call=True
)
def read_precs_data(selected_rows, data, outdir_path, meta):
    
    if not selected_rows or data is None or outdir_path is None:
        return None, None, "", [], #[]#px.scatter(title="Select a row to visualize protein quantification data")
    # outdir_path = r"C:\Users\yfu\Downloads\miss_dia_test_data\timstof_pro2\test_20250617"
    # df = pd.read_csv(Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\timstof_pro2\test_20250617\Protein.Group_Quan_maxLFQ_QT0.8.csv"))
    # meta = pd.read_csv(Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\sample_meta.csv"))
    df = pd.DataFrame(data)
    meta = pd.DataFrame(meta)
    selected_row = df.iloc[selected_rows[0]]
    dataset_name = selected_row.get("Protein.Group", "")

    # Assuming "file_name" column contains the name of the corresponding HDF file
    prec_quant_file = "Precursor.Id_quant_summ_table.parquet"
    prec_quality_file = "Precursor.Id_quality_summ_table.parquet"
    prec_quant_path = os.path.join(outdir_path, prec_quant_file)
    prec_quality_path = os.path.join(outdir_path, prec_quality_file)
    
    try:
        # Convert to DataFrame
        quant_df = pd.read_parquet(prec_quant_path, filters=[('Protein.Group', '==', dataset_name)])
        
        precursors = np.array(quant_df['Precursor.Id_MS_type'].tolist())
        precursors = np.unique(precursors)
        # precursor_options = [{'label': precursor, 'value': precursor} 
        #                      for precursor in precursors]
        
        quant_df.set_index('Precursor.Id_MS_type', drop=True, inplace=True)
        quant_df = quant_df.loc[:, meta['Sample'].values.tolist()]
        quant_df.columns = extract_unique_parts(quant_df.columns.values.tolist())
        quant_df_long = quant_df.reset_index().melt(id_vars='Precursor.Id_MS_type', var_name='X', value_name='Y')
        
        prot_title = dataset_name
        
        # read quality data
        quality_df = pd.read_parquet(prec_quality_path, filters=[('Protein.Group', '==', dataset_name)])
        quality_df.set_index('Precursor.Id_MS_type', drop=True, inplace=True)
        quality_df = quality_df.loc[:, meta['Sample'].values.tolist()]
        quality_df.columns = extract_unique_parts(quality_df.columns.values.tolist())
        quality_df_long = quality_df.reset_index().melt(id_vars='Precursor.Id_MS_type', var_name='X', value_name='Y')
        
        return (quant_df_long.to_dict("records"), 
                quality_df_long.to_dict("records"), 
                prot_title, 
                #precursor_options, 
                precursors)
    except Exception as e:
        logger.error(f"Error loading HDF file: {str(e)}")
        return None, None, "", []#, []
    

# Callback to generate figure from selected row
@app.callback(
    Output('prot-prec-quant-plot', 'figure'),
    Output('prot-prec-quality-plot', 'figure'),
    [Input('prot-quant-table', 'selected_rows'),
     Input('sub-row-means', 'value'),
     Input('show-summ', 'value'),
     #Input('prot-precursor-dropdown', 'value'),
     Input('stored-prot-prec-quant', 'data'),
     Input('stored-prot-prec-quality', 'data'),],  
    [State('stored-prot-quant', 'data'),
     State('stored-prot-quality', 'data')],
    prevent_initial_call=True  # Add this to prevent initial callback
)
def update_summ_plot(selected_rows, 
                     sub_row_means, 
                     show_summ_value, 
                     prec_quant_data, 
                     prec_quality_data,
                     quant_data, 
                     quality_data):
    ctx = callback_context
    if not ctx.triggered or not selected_rows or not quant_data or not prec_quant_data:
        return (go.Figure(layout=dict(title="Select a row to visualize protein quantification data")),
                go.Figure(layout=dict(title="Select a row to visualize protein quality data")))
    
    try:
        # Retrieve selected row data
        quant_df = pd.DataFrame(quant_data)
        selected_row_data = quant_df.iloc[selected_rows[0]]
        pg_name = selected_row_data.get("Protein.Group", "")
        
        #
        quality_df = pd.DataFrame(quality_data)
        quality_df.set_index('Protein.Group', drop=True, inplace=True)
        quality_df.columns = extract_unique_parts(quality_df.columns.values.tolist())
        
        # Make sure we're getting numerical values for the plot
        prot_quant_values = pd.to_numeric(selected_row_data.iloc[12:-1], errors='coerce').values
        prot_quality_values = pd.to_numeric(quality_df.loc[pg_name,:], errors='coerce').values
        
        # Create figure
        quant_fig = go.Figure()
        quality_fig = go.Figure()
        
        prec_quant_df = pd.DataFrame(prec_quant_data)
        #prec_quant_df["precs"] = prec_quant_df["Precursor.Id_MS_type"].apply(lambda x: x.split("_")[0])
        
        prec_quality_df = pd.DataFrame(prec_quality_data)
        #prec_quality_df["precs"] = prec_quality_df["Precursor.Id_MS_type"].apply(lambda x: x.split("_")[0])
        
        # Filter by selected precursors
        # if selected_precursors:
        #     if isinstance(selected_precursors, str):
        #         selected_precursors = [selected_precursors]
        #     prec_quant_df = prec_quant_df[prec_quant_df["precs"].isin(selected_precursors)]
        #     prec_quality_df = prec_quality_df[prec_quality_df["precs"].isin(selected_precursors)]
            
        # Plot each tract as a connected scatter plot
        for tract_id, group in prec_quant_df.groupby("Precursor.Id_MS_type"):
            if sub_row_means:
                quant_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"]-np.nanmean(group["Y"].to_numpy()), 
                    mode="lines+markers",
                    name="Precursors" if show_summ_value else tract_id,
                    line=dict(color='gray' if show_summ_value else None),
                    marker=dict(color='gray' if show_summ_value else None),
                    showlegend=not show_summ_value
                ))
            else:
                quant_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"], 
                    mode="lines+markers",
                    name="Precursors" if show_summ_value else tract_id,
                    line=dict(color='gray' if show_summ_value else None),
                    marker=dict(color='gray' if show_summ_value else None),
                    showlegend=not show_summ_value
                ))
        
        for tract_id, group in prec_quality_df.groupby("Precursor.Id_MS_type"):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else None),
                marker=dict(color='gray' if show_summ_value else None),
                showlegend=not show_summ_value
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
            title=pg_name,
            xaxis_title="Runs",
            yaxis_title="log2(intensity)",
            template="vizro",
            height=700,
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                orientation="v",
                traceorder="normal"
            ),
        )
        
        quality_fig.update_layout(
            title=pg_name,
            xaxis_title="Runs",
            yaxis_title="Quality",
            template="vizro",
            height=700,
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                orientation="v",
                traceorder="normal"
            ),
        )
        
        return quant_fig, quality_fig
    except Exception as e:
        # Return an error message in the figure
        logger.error(f"Error updating plot: {str(e)}")
        return (go.Figure(layout=dict(title=f"Error updating plot: {str(e)}")),
                go.Figure(layout=dict(title=f"Error updating plot: {str(e)}")))


@app.callback(
    Output('sequence-viewer-container', 'children'),
    Output('sequence-info', 'children'),
    Input('prot-quant-table', 'selected_rows'),
    State('stored-prot-quant', 'data')
)
def update_sequence_viewer(selected_rows, data):
    ctx = callback_context
    if not ctx.triggered or not selected_rows or not data:
        return None, None
    
    # Retrieve selected row data
    df = pd.DataFrame(data)
    row = df.iloc[selected_rows[0],:]
    
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
            return '#C50063'  # Original color
        else:
            return '#FFA500'  # Overlapping region color (adjust as needed)
    
    # Step 4: Generate coverage_data without overlaps
    coverage_data = []
    for start, end, count in segments:
        coverage_data.append({
            'start': start,
            'end': end,
            'tooltip': f"Coverage: {count} peptide(s)",
            'bgcolor': get_color(count),
            'color': 'white',
            'underscore': True
        })

    
    # Create sequence viewer with correctly formatted coverage parameter
    sequence_viewer = dashbio.SequenceViewer(
        #id='sequence-view',
        sequence=sequence,
        title=f"{row['Genes']}",
        sequenceMaxHeight='500px',
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