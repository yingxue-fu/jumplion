# builtin
from pathlib import Path
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
from plotly.subplots import make_subplots
# local
from jumplion.utils import extract_unique_parts
from jumplion.gui.app import app, to_image_config, page_sidebar_style, logger
from jumplion.gui.utils import define_species, get_valid_ratios, lfqbench_plot

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
        html.H3("Protein Quantification", style={"font-weight": "bold"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.Button('Load Quantification Table', 
                           id='load-prot-quant', n_clicks=0, color='info',
                           style={'margin-top': '10px', "textTransform": "none"}),
                html.Hr(),
                html.Label("Options for plots", style={"font-weight": "bold"}),
                html.Br(),
                html.Label("Intensity/CV Distribution plot", style={"font-weight": "bold"}),
                dcc.Dropdown(
                    id='species-options',
                    placeholder="Select a species"
                ),
                html.Br(),
                html.Label("Ratio vs. Intensity plot", style={"font-weight": "bold"}),
                dbc.Row([
                    dbc.Col(html.Label("Coefficient of variation <="), width=8),  
                    dbc.Col(html.Div([
                        dbc.Input(id="cv-filter", 
                                  type="number", 
                                  value=20, 
                                  style={"width": "80px", "height": "25px"}), 
                        html.P("%", style={"margin": "0 5px", "height": "25px"}),
                    ], style={"display": "flex", "align-items": "center"}),
                    width=4),  
                ], className="mb-1 p-0", style={"margin-top": "5px"}),
                html.Label("Select groups for comparison", style={'marginTop': '5px'}),
                dcc.Dropdown(id='group-A', placeholder="Group A"),
                dcc.Dropdown(id='group-B', placeholder="Group B", style={'marginTop': '5px'}),
                html.Label("Log2(Ratio) Range:", style={'marginTop': '5px'}),
                dcc.Slider(
                    id='y-span',
                    min=1,
                    max=5,
                    step=1,
                    value=3,
                    marks={i: str(i) for i in range(1, 6)}
                ),
                html.Br(),
                html.Label("Quantity/Quality Summarization plot", style={"font-weight": "bold", 'marginBottom': '5px'}),
                dbc.Switch(label='Show raw precursor quantity (before normalization)', id='show-raw', value=False),
                dbc.Switch(label='Show relative ratio of quantity', id='sub-row-means', value=False),
                dbc.Switch(label='Show summarized quantity/quality', id='show-summ', value=False),
                html.Hr(),
                html.Label("Other plot options", style={"font-weight": "bold", 'marginBottom': '5px'}),
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
                    value=0.2,
                    marks={i/10: str(i/10) for i in range(1, 11)}
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
                    dbc.CardHeader(html.H5("Protein quantification table", style={"font-weight": "bold"})),
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
                                'backgroundColor': 'lightgray', #'lightgray',
                                'fontWeight': 'bold'},
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            page_size=10,
                            page_action="native",
                        ),
                        html.Div(id="prot-quant-total-count", 
                                 style={
                                    'position': 'absolute',
                                    'bottom': '25px',
                                    'left': '15px',
                                    'fontSize': '14px',
                                    'color': 'black',
                                    'fontFamily': 'Arial, sans-serif',
                                    'backgroundColor': 'rgba(255,255,255,0.9)',
                                    'padding': '0px',
                                    'borderRadius': '0px',
                                    'zIndex': '10' # controls the stacking order of elements
                                })
                    ], style={'position': 'relative'})
                ], style={"width": "97%"})
            ], className="dbc dbc-row-selectable"),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Protein quantification plots", style={"font-weight": "bold"})),
                    dbc.CardBody(
                        [
                            dbc.Tabs(
                                id="tabs",
                                active_tab="tab-1",
                                children=[
                                    dbc.Tab(label="Intensity Distribution", 
                                            tab_id="tab-1", label_style=tab_label_style),
                                    dbc.Tab(label="CV Distribution", 
                                            tab_id="tab-2", label_style=tab_label_style),
                                    dbc.Tab(label="CV vs. Intensity", 
                                            tab_id="tab-3", label_style=tab_label_style),
                                    dbc.Tab(label="Ratio vs. Intensity", 
                                            tab_id="tab-4", label_style=tab_label_style),
                                    dbc.Tab(label="Quantity Summarization", 
                                            tab_id="tab-5", label_style=tab_label_style),
                                    dbc.Tab(label="Quality Summarization", 
                                            tab_id="tab-6", label_style=tab_label_style),
                                    dbc.Tab(label="Coverage Information", 
                                            tab_id="tab-7", label_style=tab_label_style),
                                ],
                                style=tab_bar_style
                            ),
                            html.Div(
                                [   
                                    # Tab 1:
                                    html.Div(
                                        [
                                            dcc.Graph(id="intensity-violin-plot", style={'height': '500px'}),
                                        ],    
                                        id="tab-1-content",
                                    ),
                                    # Tab 2:
                                    html.Div(
                                        [
                                            html.Div([
                                                dcc.Graph(id='cv-violin-plot', config=to_image_config, style={'height': '500px'})
                                            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                            
                                            html.Div([
                                                dcc.Graph(id='cv-bar-plot', config=to_image_config, style={'height': '500px'})
                                            ], style={'width': '55%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        ],    
                                        id="tab-2-content",
                                    ),
                                    # Tab 3:
                                    html.Div(
                                        [
                                            dcc.Graph(id="cv-intensity-plot", style={'height': '500px'}),
                                        ],    
                                        id="tab-3-content",
                                    ),
                                    # Tab 4:
                                    html.Div(
                                        [
                                            html.Div([
                                                dcc.Graph(id='ratio-intensity-plot', config=to_image_config, style={'height': '500px'})
                                            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                            
                                            html.Div([
                                                dcc.Graph(id='ratio-density-plot', config=to_image_config, style={'height': '500px'})
                                            ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}),
                                        ], 
                                        id="tab-4-content",
                                    ),
                                    # Tab 5:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quant-plot", config=to_image_config, style={'height': '500px'}),
                                            dbc.Button("Check Precursors", id="check-prec-quantity", 
                                                       color="info",  
                                                       style={"font-weight": "bold",  
                                                              "marginTop": "5px",
                                                              "textTransform": "none"}),
                                        ],    
                                        id="tab-5-content",
                                    ),
                                    # Tab 6:
                                    html.Div(
                                        [
                                            dcc.Graph(id="prot-prec-quality-plot", config=to_image_config, style={'height': '500px'}),
                                            dbc.Button("Check Precursors", id="check-prec-quality", 
                                                       color="info",  
                                                       style={"font-weight": "bold", 
                                                              "marginTop": "5px",
                                                              "textTransform": "none"}),
                                        ],    
                                        id="tab-6-content",
                                    ), 
                                    # Tab 7:
                                    html.Div(
                                        [
                                            html.Div(id='sequence-info'),
                                            html.Div(id='sequence-viewer-container'),
                                        ],    
                                        id="tab-7-content",
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
       

# Callback to read the selected CSV file
@app.callback(
    [Output('prot-quant-table', 'data'),
     Output('prot-quant-table', 'columns'),
     Output('species-options', 'options'),
     Output('stored-prot-quant', 'data'),
     Output('stored-prot-quality', 'data'),
     Output('stored-meta', 'data')],
    Input('load-prot-quant', 'n_clicks'),
    State('stored-outdir', 'data'),
    State('meta-path', 'value'),
    prevent_initial_call=True
)
def load_quant_table(n_clicks, outdir_path, meta_path):
    
    if not outdir_path or not meta_path:
        return None, [], [], None, None, {}
    
    out_int_dir = Path(outdir_path) / 'intermediate'
    out_pub_dir = Path(outdir_path) / 'publication'
    quant_csv_path = out_pub_dir / 'protein.group_quantity_matrix_all.csv'
    quality_csv_path = out_pub_dir / "protein.group_quality_matrix_all.csv"
    prot_pep_info_path = out_int_dir / "prot_prec_pep_info.parquet"
    
    try:
        quant_df = pd.read_csv(quant_csv_path)
        meta_df = pd.read_csv(meta_path)
        print(f"Number of proteins in the initial table: {len(quant_df)}")
        
        # Make the samples the same order as in the meta file
        sample_names = meta_df['Sample'].values.tolist()
        quant_df = quant_df.loc[:, quant_df.columns[0:3].values.tolist() + sample_names]
        
        # Simplify the sample name
        simple_sample_names = extract_unique_parts(sample_names)
        quant_df.columns = (
            quant_df.columns[0:3].values.tolist() + 
            simple_sample_names
        )
        meta_df['Sample_simp'] = simple_sample_names
        
        # Calculate CV
        groups = sorted(meta_df['Group'].dropna().unique().tolist())
        for group in groups:
            cols_group = meta_df.loc[meta_df['Group'] == group, 'Sample_simp'].values
            group_df = quant_df[cols_group]
            quant_df[f'Mean_{group}'] = group_df.mean(axis=1)
            quant_df[f'CV_{group}'] = (group_df.std(axis=1) / quant_df[f'Mean_{group}']) * 100
        
        # Define protein species
        quant_df['Species'] = quant_df['Protein.Names'].apply(define_species)
        species_count = quant_df['Species'].value_counts(dropna=False) 
        if any(species_count <= 10):
            print("Removed proteins of the following species due to n <= 10")
            print(species_count[species_count <= 10].to_string())
        species_list = species_count[species_count > 10].index.tolist()
        species_options = [{'label': s, 'value': s} for s in species_list if s is not None ]
        quant_df = quant_df[quant_df['Species'].isin(species_list)]
        
        # Combine with protein coverage info
        prot_pep_df = pd.read_parquet(prot_pep_info_path)
        quant_df = prot_pep_df.merge(quant_df, how='right')
        cols_to_show = ["Protein.Group", "Protein.Names", "Genes", "n.Precursors", "n.Peptides"]
        quant_df_to_show = quant_df[cols_to_show+simple_sample_names]
        quant_df_to_show[simple_sample_names] = quant_df_to_show[simple_sample_names].round(3)
        #
        quality_df = pd.read_csv(quality_csv_path)
        quality_df = quality_df.loc[:, ['Protein.Group'] + meta_df['Sample'].values.tolist()]
        
        return (quant_df_to_show.to_dict('records'), 
                [{'name': col, 'id': col} for col in quant_df_to_show.columns], 
                species_options,
                quant_df.to_dict('records'), 
                quality_df.to_dict('records'),
                meta_df.to_dict('records'))

    except Exception as e:
        logger.error(f"Error loading quantification table: {str(e)}")
        return None, [], [], None, None, {}


@app.callback(
    Output('prot-quant-total-count', 'children'),
    [Input('prot-quant-table', 'derived_virtual_data'),
     Input('stored-prot-quant', 'data')]
)
def update_table_footer(filtered_data, data):
    if not filtered_data or not data:
        return "Total: 0 rows"
    
    total_original = len(data)
    total_filtered = len(filtered_data)
    
    if total_filtered == total_original:
        return f"Total: {total_original} rows"
    else:
        return f"{total_filtered} of {total_original} rows"
    
    
@app.callback(
    Output('intensity-violin-plot', 'figure'),
    Input('stored-prot-quant', 'data'),
    Input('species-options', 'value'),
    Input('stored-meta', 'data'),
    Input('point-size', 'value'),
    Input('point-opacity', 'value'),
)
def update_intensity_violin_plot(data, species_input, meta_data, point_cex, opacity):
    if data is None:
        return dash.no_update

    # Load data
    df = pd.DataFrame(data)
    meta = pd.DataFrame(meta_data)
    value_columns = meta['Sample_simp']
    
    if not species_input:
        df_filtered = df.copy()
        df_filtered['Species'] = 'ALL'
    else:
        df_filtered = df[df['Species'] == species_input]

        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')

    # Melt data for plotly
    df_melted = df_filtered.melt(id_vars=['Species'], value_vars=value_columns,
                                 var_name='Sample', value_name='Intensity')
    df_melted['log2Intensity'] = np.log2(df_melted['Intensity'])

    # Create figure
    fig = go.Figure()
    fig = px.violin(df_melted, 
                    x='Sample', 
                    y='log2Intensity', 
                    color='Species', 
                    color_discrete_map=colors,
                    box=True, 
                    points='outliers')
    
    fig.update_traces(
        points='outliers', 
        marker=dict(opacity=opacity, size=point_cex*10),
        selector=dict(type="violin")  # Apply only to box traces
    )
    
    fig.update_layout(
        title='', 
        xaxis_title="Runs",
        yaxis_title="Log2(Intensity)",
        violingap=0,
        template="vizro",
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
    Output('cv-violin-plot', 'figure'),
    Output('cv-bar-plot', 'figure'),
    Input('stored-prot-quant', 'data'),
    Input('species-options', 'value'),
    Input('point-size', 'value'),
    Input('point-opacity', 'value'),
)
def update_CV_plots(data, species_input, point_cex, opacity):
    if data is None:
        return dash.no_update

    # Load data
    df = pd.DataFrame(data)
    CV_columns = [col for col in df.columns if 'CV_' in col]
    
    if not species_input:
        df_filtered = df.copy()
        df_filtered['Species'] = 'ALL'
    else:
        df_filtered = df[df['Species'] == species_input]

        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')

    # Melt data for plotly
    df_melted = df_filtered.melt(id_vars=['Species'], value_vars=CV_columns,
                                 var_name='Group', value_name='CV')
    df_melted['Group'] = df_melted['Group'].str.replace('CV_', '')
    
    # Create figure
    fig_violin = go.Figure()
    fig_violin = px.violin(df_melted, 
                           x='Group', 
                           y='CV', 
                           color='Species', 
                           color_discrete_map=colors,
                           box=True, 
                           points='outliers')

    fig_violin.update_layout(
        title='', 
        xaxis_title="Group",
        yaxis_title="CV",
        violingap=0,
        template="vizro",
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    fig_violin.update_traces(
        points='outliers', 
        marker=dict(opacity=opacity, size=point_cex*10),
        selector=dict(type="violin")  # Apply only to box traces
    )
    
    # Calculate count statistics for each variable
    count_stats = []
    groups = df_melted['Group'].unique().tolist()
    for group in groups:
        var_data = df_melted[df_melted['Group'] == group]['CV']
        total_count = len(var_data)
        
        stats = {
            'Group': group,
            'Total': total_count,
            'CV<=20': (var_data <= 20).sum(),
            'CV<=10': (var_data <= 10).sum(),
            'CV<=5': (var_data <= 5).sum()
        }
        count_stats.append(stats)
    
    # Convert to DataFrame for easier plotting
    count_df = pd.DataFrame(count_stats)
    for group in groups:
        total = count_df[count_df['Group'] == group]['Total'].iloc[0]
        count_df.loc[count_df['Group'] == group, 'Total_pct'] = 100.0
        count_df.loc[count_df['Group'] == group, 'CV<=20_pct'] = (count_df.loc[count_df['Group'] == group, 'CV<=20'].iloc[0] / total) * 100
        count_df.loc[count_df['Group'] == group, 'CV<=10_pct'] = (count_df.loc[count_df['Group'] == group, 'CV<=10'].iloc[0] / total) * 100
        count_df.loc[count_df['Group'] == group, 'CV<=5_pct'] = (count_df.loc[count_df['Group'] == group, 'CV<=5'].iloc[0] / total) * 100
    
    # Melt the count data for plotting
    count_melted = count_df.melt(id_vars=['Group'], 
                                 value_vars=['Total', 'CV<=20', 'CV<=10', 'CV<=5'],
                                 var_name='CV_category', 
                                 value_name='Count')
    
    # Melt the percentage data and merge
    pct_melted = count_df.melt(id_vars=['Group'], 
                              value_vars=['Total_pct', 'CV<=20_pct', 'CV<=10_pct', 'CV<=5_pct'],
                              var_name='CV_category_pct', 
                              value_name='Percentage')
    pct_melted['CV_category'] = pct_melted['CV_category_pct'].str.replace('_pct', '')
    
    # Merge count and percentage data
    count_melted = count_melted.merge(pct_melted[['Group', 'CV_category', 'Percentage']], 
                                      on=['Group', 'CV_category'])
    
    # Create text annotations
    count_melted['text'] = count_melted['Count'].astype(str) + '\n(' + count_melted['Percentage'].round(1).astype(str) + '%)'
    
    # Define color map with gradually darker grays
    gray_colors = {
        'Total': '#D3D3D3',          # Light gray
        'CV<=20': '#A9A9A9',         # Dark gray
        'CV<=10': '#696969',         # Dim gray
        'CV<=5': '#2F2F2F'           # Very dark gray
    }
    
    # Create bar plot
    fig_bar = px.bar(count_melted, 
                     x='Group', 
                     y='Count', 
                     color='CV_category',
                     color_discrete_map=gray_colors,
                     barmode='group',
                     #title='Number of proteins by CV Category',
                     text='text',
                     labels={'Count': 'Number of proteins', 'CV_category': 'CV Category'})
    fig_bar.update_traces(textposition='inside', textfont_size=14)
    
    # Update layout for better appearance
    fig_bar.update_layout(
        xaxis_title='Group',
        yaxis_title='Number of proteins',
        legend_title='',
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    
    return fig_violin, fig_bar


@app.callback(
    Output('cv-intensity-plot', 'figure'),
    Input('stored-prot-quant', 'data'),
    Input('species-options', 'value'),
    Input('point-size', 'value'),
    Input('point-opacity', 'value'),
    State('stored-meta', 'data'),
)
def update_CV_intensity_plots(data, species_input, point_cex, opacity, meta_data):
    if data is None:
        return dash.no_update

    # Load data
    df = pd.DataFrame(data)
    meta_df = pd.DataFrame(meta_data)
    groups = sorted(meta_df['Group'].unique().tolist())
    
    if not species_input:
        df_filtered = df.copy()
        df_filtered['Species'] = 'ALL'
    else:
        df_filtered = df[df['Species'] == species_input]
        if df_filtered.empty:
            return px.scatter(title='No matching species found in your input.')
    species_final = df_filtered['Species'].unique()[0]
    color = colors[species_final]
    
    # Create figure
    n_groups = len(groups)
    cols = min(3, n_groups)  # Max 3 columns
    rows = int(np.ceil(n_groups / cols))
    
    # Create subplots
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f'{species_final}: Group {group} (n = {df_filtered.shape[0]})' for group in groups],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # Add scatter plot for each group
    for i, group in enumerate(groups):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        x_col = f'Mean_{group}'
        y_col = f'CV_{group}'
        
        if x_col in df_filtered.columns and y_col in df_filtered.columns:
            # Remove NaN values for this group
            mask = df_filtered[x_col].notna() & df_filtered[y_col].notna()
            x_data = np.log2(df_filtered.loc[mask, x_col])
            y_data = df_filtered.loc[mask, y_col]
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    #name=f'{species}: Group {group} (n = {df_filtered.shape[0]})',
                    marker=dict(
                        size=point_cex*10,
                        opacity=opacity,
                        color=color
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Update axes labels for this subplot
            fig.update_xaxes(title_text='Mean Intensity (log2)', row=row, col=col)
            fig.update_yaxes(title_text='CV', range=[0, 50], row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=500 * rows
    )
    
    return fig

@app.callback(
    Output("tab-1-content", "style"),
    Output("tab-2-content", "style"),
    Output("tab-3-content", "style"),
    Output("tab-4-content", "style"),
    Output("tab-5-content", "style"),
    Output("tab-6-content", "style"),
    Output("tab-7-content", "style"),
    Input("tabs", "active_tab")
)
def toggle_tab(tab):
    styles = {"display": "none"}
    if tab == "tab-1":
        return {}, styles, styles, styles, styles, styles, styles
    elif tab == "tab-2":
        return styles, {}, styles, styles, styles, styles, styles
    elif tab == "tab-3":
        return styles, styles, {}, styles, styles, styles, styles
    elif tab == "tab-4":
        return styles, styles, styles, {}, styles, styles, styles
    elif tab == "tab-5":
        return styles, styles, styles, styles, {}, styles, styles
    elif tab == "tab-6":
        return styles, styles, styles, styles, styles, {}, styles
    else:
        return styles, styles, styles, styles, styles, styles, {}

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
     Output('ratio-density-plot', 'figure'),
     Output('stored-valid-prots', 'data')],
    [Input('cv-filter', 'value'),
     Input('group-A', 'value'),
     Input('group-B', 'value'),
     Input('y-span', 'value'),
     Input('point-size', 'value'),
     Input('point-opacity', 'value'),
     State('stored-prot-quant', 'data'),
     State('stored-meta', 'data')],
    prevent_initial_call=True
)
def update_ratio_intensity_plot(cv_filter, group_A, group_B, y_span, point_size, point_opacity, data, meta):
    if data is None or group_A is None or group_B is None:
        return dash.no_update, dash.no_update, {}
    
    # Load data
    df = pd.DataFrame(data)
    meta = pd.DataFrame(meta) 
    value_columns = extract_unique_parts(meta['Sample'].to_list())
    
    print(f"Comparing between {group_A} and {group_B}")
    cv_columns = [f'CV_{group_A}', f'CV_{group_B}']
    for cv_col in cv_columns:
        print(f"Filtering proteins based on the {cv_col} (<={cv_filter})")
        print(f"Before: {df.shape[0]}")
        df = df[df[cv_col] <= cv_filter]
        print(f"After: {df.shape[0]}")
    
    protein_tags = df["Protein.Group"].values.tolist()
    protein_species = df['Species'].values.tolist()
    species_list = df['Species'].unique().tolist()
    
    cols_A = meta.index[meta['Group'] == group_A]
    cols_B = meta.index[meta['Group'] == group_B]  
    
    # Create LFQBench plots
    mat = np.log2(df[value_columns])
    valid_prots = get_valid_ratios(mat, protein_tags, protein_species, species_list, 
                                   sample_groups=[group_A, group_B], 
                                   cols_A=cols_A, cols_B=cols_B)
    scatter_fig, density_fig = lfqbench_plot(
        valid_prots, species_list, 
        colors=colors, span=y_span, point_cex=point_size, 
        sample_groups=[group_A, group_B], title="",
        opacity=point_opacity
    )
    
    scatter_fig.update_layout(clickmode='event+select')
    
    valid_prots_all = [item for sublist in valid_prots.values() for item in sublist]
    return scatter_fig, density_fig, {tag: (x, y) for tag, x, y in valid_prots_all}


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

    out_pub_dir = Path(outdir_path) / 'publication'
    prec_quant_path = out_pub_dir / "precursor_quantity_summ_table.parquet"
    prec_quant_norm_path = out_pub_dir / "precursor_quantity_summ_table_norm.parquet"
    prec_quality_path = out_pub_dir / "precursor_quality_summ_table.parquet"
    
    try:
        # read quant data
        quant_df = pd.read_parquet(prec_quant_path, filters=[('Protein.Group', '==', selected_prot)])
        precursors = [precursor.split("_")[0] for precursor in quant_df['Precursor.Id_MS_type'].tolist()]
        precursors = np.unique(np.array(precursors))
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
     State('stored-prot-quality', 'data'),
     State('stored-meta', 'data')],
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
                     quality_data, 
                     meta_data):
    if not ctx.triggered or not selected_prot or not quant_data or not prec_quant_data:
        return (go.Figure(layout=dict(title="Select a protein to visualize protein quantification data")),
                go.Figure(layout=dict(title="Select a protein to visualize protein quality data")))
    
    try:
        meta_df = pd.DataFrame(meta_data)
        quant_df = pd.DataFrame(quant_data)
        quality_df = pd.DataFrame(quality_data)
        quality_df.set_index('Protein.Group', drop=True, inplace=True)
        quality_df.columns = extract_unique_parts(quality_df.columns.values.tolist())
        # Retrieve selected row data
        value_cols = meta_df['Sample_simp'].to_list()
        prot_quant_values = np.log2(quant_df.loc[quant_df["Protein.Group"] == selected_prot, value_cols].values[0])
        prot_quality_values = quality_df.loc[selected_prot, value_cols].values
        
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
            group["Y"] = np.log2(group["Y"])
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
                    showlegend=not show_summ_value,
                    opacity=0.7 if show_summ_value else 1.0
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
            quant_fig.add_trace(go.Scatter(
                x=value_cols, 
                y=summ_values, 
                mode="lines+markers",
                name="Summarization",
                line=dict(color='blue', width=2),
                marker=dict(color='blue', size=8)
            ))
            
            quality_fig.add_trace(go.Scatter(
                x=value_cols, 
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
            xaxis=dict(domain=[0, 0.7]),  # Plot uses only 70% of width
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
            xaxis=dict(domain=[0, 0.7]),  # Plot uses only 70% of width
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
                        style={'height': '500px',
                               'fontSize': '20px', 
                               'marginTop': '15px',
                               'marginLeft': '20px'}), 
                None)
    
    # Retrieve selected row data
    df = pd.DataFrame(data)
    if 'Peptide.Positions' not in df.columns:
        return (html.Div("No coverage information is found for the selected protein.", 
                        style={'height': '500px',
                               'fontSize': '20px', 
                               'marginTop': '15px',
                               'marginLeft': '20px'}), 
                None)
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