# builtin
from pathlib import Path
from typing import List, Dict, Any#, Tuple, Optional
# external
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table#, callback_context
from dash_bootstrap_templates import load_figure_template
# local
from jumpdiaq.utils import extract_unique_parts
from jumpdiaq.gui.app import app, page_sidebar_style, logger


load_figure_template(["vizro", "vizro_dark"])
ms1_colors = dict(zip([f'M+{i}' for i in range(9)], px.colors.qualitative.T10[0:9]))
ms2_colors = px.colors.qualitative.Bold

sidebar = html.Div(
    [
        html.H3("Precursor Quantification", style={"font-weight": "bold"}),
        html.Hr(),
        dbc.Nav(
            [   
                html.Label('Select a precursor', style={"font-weight": "bold"}),
                dcc.Dropdown(id='precursor-dropdown', placeholder="Select a Precursor"),
                html.Label(id='protein-label', style={'marginTop': '5px'}),
                html.Br(),
                html.Label("Precursor summarization", style={"font-weight": "bold"}),
                dbc.Switch(label='Show relative ratio of quantity', id='sub-row-means-xic', value=False),
                dbc.Switch(label='Show summarized quantity/quality', id='show-prec-summ', value=False),
                html.Br(),
                html.Label("Show precursor XICs in a run", style={"font-weight": "bold"}),
                dcc.Dropdown(id='run-dropdown', placeholder="Select a Run"),
                dbc.Switch(label='Show Interpolated MS1 XICs', id='show-ms1-inter', value=False, 
                           className="d-inline-block", style={'marginTop': '5px'}),
                html.Hr(),
                dbc.Button("Go back to Proteins", id="view-prot-prec", 
                           color="info",  
                           style={"font-weight": "bold", 
                                  "width": "150px", 
                                  "marginTop": "5px",
                                  "textTransform": "none"}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=page_sidebar_style,
)

tabs_styles = {
    'margin-top': '50px'
}

tab_label_style = {
    'fontSize': '14px',
    "textTransform": "none",  # Prevent all caps
    "textAlign": "center"
}

layout = html.Div(children=[
    dcc.Store(id='stored-precquant'),  # Store the protein quantification data
    dcc.Store(id='stored-precquality'),
    dcc.Store(id='stored-precMS1quant'),
    dcc.Store(id='stored-precMS1quality'),
    dcc.Store(id='stored-precMS2quant'),
    dcc.Store(id='stored-precMS2quality'),
    dcc.Store(id='stored-report'),  # Added for storing DataFrame
    dbc.Row([
        dbc.Col(sidebar, width=4),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Precursor summarization', children=[
                        dbc.Row([
                            dbc.Col(html.Div(id='show-precid-c', style={'fontSize': '20px'}), width=11)
                        ], style={'margin-top': '20px'}),
                        html.Div([   
                            dcc.Graph(id="prec-quant-plot"),
                            dcc.Graph(id="prec-quality-plot"),
                            ], className="p-2")
                ], label_style=tab_label_style),
                dbc.Tab(label='Precursor XICs', children=[
                    dbc.Row([
                        dbc.Col(html.Div(id='show-precid-i', style={'fontSize': '20px'}), width=11)
                    ], style={'margin-top': '20px'}),
                    dbc.Row([
                        dbc.Col(html.Div(id='show-run-info', style={'fontSize': '16px', 'color': 'green'}), width=11)
                    ], style={'margin-top': '5px'}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='xic-plot-ms1'), width=8),
                        dbc.Col(dcc.Graph(id='xic-plot-ms1-c'), width=4)
                    ], style={'margin-top': '5px', 'margin-right': '10px'}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='xic-plot-ms2'), width=8),
                        dbc.Col(dcc.Graph(id='xic-plot-ms2-c'), width=4)
                    ], style={'margin-top': '5px', 'margin-right': '10px'}),
                    dbc.Row([
                        dbc.Col(dash_table.DataTable(
                            id='xic-info-table',
                            columns=[{'name': col, 'id': col} for col in ["XIC Type", 
                                                                          "XIC Name",
                                                                          "XIC m/z", 
                                                                          "XIC Rank Score",
                                                                          "XIC Corr", 
                                                                          "XIC Quant (Raw)", 
                                                                          "XIC Quant (Adjusted)"]],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={'fontFamily': 'Arial, sans-serif',
                                        'fontSize': '14px',
                                        'textAlign': 'left', 
                                        'padding': '5px'},
                            style_header={'fontWeight': 'bold'},
                            page_size=10,
                        ), width=11, className="mx-auto"),
                    ], style={'margin-top': '15px', 'margin-bottom': '20px'}),
                    ], label_style=tab_label_style)
                ], style=tabs_styles),
        ], width=8)
    ])
], className="dbc dbc-row-selectable")


def get_xic_path(data_path: Path, run_name: str) -> Path:
    """
    Construct absolute path for XIC file
    """
    try:
        data_dir = data_path#.parent
        xic_file_path = (data_dir / "xic_files" / f"{run_name}.xic.parquet").resolve()
        return xic_file_path
    except Exception as e:
        logger.error(f"Error constructing XIC path: {e}")
        raise

def string2float(float_strings: List[str], decimal: int) -> List[str]:
    """Safely convert string numbers to floats with specified decimal places."""
    float_list = []
    for value in float_strings:
        try:
            float_list.append(float(value))
        except (ValueError, TypeError):
            float_list.append(0.0)  # Default value for invalid numbers
    return [f"{value:.{decimal}f}" for value in float_list]


@app.callback(
    Output('protein-label', 'children'),
    Input('selected-prot', 'data'),
)
def update_protein_label(protein_name):
    return f"Protein: {protein_name}"

@app.callback(
    Output('precursor-dropdown', 'options'),
    Input('stored-precursor-options', 'data')
)
def load_options(stored_options):
    if stored_options is None:
        return []
    
    precursor_options = [{'label': precursor, 'value': precursor} 
                         for precursor in stored_options]
    return precursor_options

@app.callback(
    [Output('show-precid-c', 'children'),
     Output('stored-precquant', 'data'),
     Output('stored-precquality', 'data'),
     Output('stored-precMS1quant', 'data'),
     Output('stored-precMS1quality', 'data'),
     Output('stored-precMS2quant', 'data'),
     Output('stored-precMS2quality', 'data')],
    Input('precursor-dropdown', 'value'),
    State('stored-outdir', 'data'),
    prevent_initial_call=True
)
def read_precs_data(selected_precursor, outdir_path):
    if not selected_precursor or outdir_path is None:
        return None, None, None, None, None, None, None
    
    out_prec_dir = Path(outdir_path) / 'intermediate' / 'prec_quant'
    #
    prec_quant_file = out_prec_dir / "precursor_quantity_ms1_ms2_summ_table.parquet"
    prec_quality_file = out_prec_dir / "precursor_quality_ms1_ms2_summ_table.parquet"
    #
    ms1_quant_file = out_prec_dir / "Precursor.Id_MS1.XIC.Quant.Adjusted_mapping.parquet"
    ms1_quality_file = out_prec_dir / "Precursor.Id_MS1.XIC.Corr.Raw_mapping.parquet"
    #
    ms2_quant_file = out_prec_dir / "Precursor.Id_MS2.XIC.Quant.Raw_mapping.parquet"
    ms2_quality_file = out_prec_dir / "Precursor.Id_MS2.XIC.Corr.Raw_mapping.parquet"
    
    precursor_id = selected_precursor
    title = html.Div([f"     {precursor_id}"])
    try:
        prec_quant_df = pd.read_parquet(prec_quant_file, filters=[('Precursor.Id', '==', precursor_id)])
        prec_quality_df = pd.read_parquet(prec_quality_file, filters=[('Precursor.Id', '==', precursor_id)])
        
        ms1_quant_df = pd.read_parquet(ms1_quant_file, filters=[('target_id', '==', precursor_id)])
        ms1_quant_df = ms1_quant_df[ms1_quant_df['source_quantity'] > 0]
        ms1_quant_df['source_quantity'] = np.log2(ms1_quant_df['source_quantity'])
        ms1_quant_df['samples'] = extract_unique_parts(ms1_quant_df['samples'].values.tolist())
        
        ms1_quality_df = pd.read_parquet(ms1_quality_file, filters=[('target_id', '==', precursor_id)])
        ms1_quality_df = ms1_quality_df[~(ms1_quality_df['source_quality'] == 0)]
        ms1_quality_df['samples'] = extract_unique_parts(ms1_quality_df['samples'].values.tolist())
        
        ms2_quant_df = pd.read_parquet(ms2_quant_file, filters=[('target_id', '==', precursor_id)])
        ms2_quant_df = ms2_quant_df[ms2_quant_df['source_quantity'] > 0]
        ms2_quant_df['source_quantity'] = np.log2(ms2_quant_df['source_quantity'])
        ms2_quant_df['samples'] = extract_unique_parts(ms2_quant_df['samples'].values.tolist())
        
        ms2_quality_df = pd.read_parquet(ms2_quality_file, filters=[('target_id', '==', precursor_id)])
        ms2_quality_df = ms2_quality_df[~(ms2_quality_df['source_quality'] == 0)]
        ms2_quality_df['samples'] = extract_unique_parts(ms2_quality_df['samples'].values.tolist())
        
        return (title, 
                prec_quant_df.to_dict("records"), 
                prec_quality_df.to_dict("records"), 
                ms1_quant_df.to_dict("records"), 
                ms1_quality_df.to_dict("records"), 
                ms2_quant_df.to_dict("records"), 
                ms2_quality_df.to_dict("records"),
                )
    except Exception as e:
        logger.error(f"Error loading parquet file: {str(e)}")
        return None, None, None, None, None, None, None

@app.callback(
    Output('prec-quant-plot', 'figure'),
    Output('prec-quality-plot', 'figure'),
    Input('sub-row-means-xic', 'value'),
    Input('show-prec-summ', 'value'),
    Input('stored-precquant', 'data'),
    Input('stored-precquality', 'data'),
    Input('stored-precMS1quant', 'data'),
    Input('stored-precMS1quality', 'data'),
    Input('stored-precMS2quant', 'data'),
    Input('stored-precMS2quality', 'data'), 
    State('stored-meta', 'data'),
    prevent_initial_call=True  
)
def update_summ_plot(sub_row_means_xic,
                     show_summ_value, 
                     prec_quant_data, 
                     prec_quality_data,
                     ms1_quant_data, 
                     ms1_quality_data,
                     ms2_quant_data, 
                     ms2_quality_data,
                     meta):
    #ctx = callback_context
    if not prec_quant_data or not ms1_quant_data: # not ctx.triggered or
        return (go.Figure(layout=dict(title="Select a precursor to visualize its quantification data")),
                go.Figure(layout=dict(title="Select a precursor to visualize its quality data")))
    
    try:
        meta = pd.DataFrame(meta)
        meta_samples = meta['Sample'].values.tolist()
        prec_quant_df = pd.DataFrame(prec_quant_data)
        prec_quant_values = prec_quant_df.loc[:, meta_samples+['MS_type']]
        prec_quant_values = prec_quant_values.melt(id_vars='MS_type', var_name='X', value_name='Y')
        prec_quant_values['X'] = extract_unique_parts(prec_quant_values['X'].values.tolist())
        
        prec_quality_df = pd.DataFrame(prec_quality_data)
        prec_quality_values = prec_quality_df.loc[:, meta_samples+['MS_type']]
        prec_quality_values = prec_quality_values.melt(id_vars='MS_type', var_name='X', value_name='Y')
        prec_quality_values['X'] = extract_unique_parts(prec_quality_values['X'].values.tolist())
        
        # Retrieve selected row data
        ms1_quant_df = pd.DataFrame(ms1_quant_data)
        ms1_quality_df = pd.DataFrame(ms1_quality_data)
        
        ms2_quant_df = pd.DataFrame(ms2_quant_data)
        ms2_quality_df = pd.DataFrame(ms2_quality_data)
        
        def pivot_order_melt(df, value_type, samples):
            # Pivot to wide format
            df_pivot = df.pivot(index='source_id', 
                                columns='samples', 
                                values=f'source_{value_type}')
            # Add missing samples as NaN
            for sample in samples:
                if sample not in df_pivot.columns:
                    df_pivot[sample] = np.nan
            # Reorder columns
            df_pivot = df_pivot[samples]
            # Convert to long format
            df_long = df_pivot.reset_index().melt(id_vars='source_id', 
                                                  var_name='X', 
                                                  value_name='Y')
            return df_long
        
        # Create figure
        quant_fig = go.Figure()
        quality_fig = go.Figure()
        
        # Plot each tract as a connected scatter plot
        meta_samples = extract_unique_parts(meta_samples)
        
        ms1_quant_df = pivot_order_melt(ms1_quant_df, 'quantity', meta_samples)
        for tract_id, group in ms1_quant_df.groupby("source_id"):
            x=group["X"]
            if sub_row_means_xic:
                y=group["Y"]-np.nanmean(group["Y"].to_numpy())
                opacity=0.7
            else:
                y=group["Y"]
                opacity=1.0
                
            quant_fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode="lines+markers",
                name="MS1 XICs" if show_summ_value else tract_id,
                line=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                marker=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                showlegend=not show_summ_value,
                opacity=opacity
            ))

        ms2_quant_df = pivot_order_melt(ms2_quant_df, 'quantity', meta_samples)
        for i, (tract_id, group) in enumerate(ms2_quant_df.groupby("source_id")):
            x=group["X"]
            if sub_row_means_xic:
                y=group["Y"]-np.nanmean(group["Y"].to_numpy())
                opacity=0.7
            else:
                y=group["Y"]
                opacity=1.0
                
            quant_fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode="lines+markers",
                name="MS2 XICs" if show_summ_value else tract_id,
                line=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                marker=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                showlegend=not show_summ_value,
                opacity=opacity
            ))
        
        ms1_quality_df = pivot_order_melt(ms1_quality_df, 'quality', meta_samples)
        for tract_id, group in ms1_quality_df.groupby("source_id"):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                marker=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                showlegend=not show_summ_value,
                opacity=0.7 if show_summ_value else 1.0
            ))
        ms2_quality_df = pivot_order_melt(ms2_quality_df, 'quality', meta_samples)
        for i, (tract_id, group) in enumerate(ms2_quality_df.groupby("source_id")):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                marker=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                showlegend=not show_summ_value,
                opacity=0.7 if show_summ_value else 1.0
            ))
        
        color_map = {"MS1": "purple", "MS2": "blue"}
        if show_summ_value:
            for tract_id, group in prec_quant_values.groupby("MS_type"):
                x=group["X"]
                if sub_row_means_xic:
                    y=group["Y"]-np.nanmean(group["Y"].to_numpy())
                    opacity=0.7
                else:
                    y=group["Y"]
                    opacity=1.0
                    
                quant_fig.add_trace(go.Scatter(
                    x=x, 
                    y=y, 
                    mode="lines+markers",
                    name=tract_id,
                    line=dict(color=color_map.get(tract_id, "gray"), width=2),
                    marker=dict(color=color_map.get(tract_id, "gray"), size=8),
                ))
            
            for tract_id, group in prec_quality_values.groupby("MS_type"):
                quality_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"], 
                    mode="lines+markers",
                    name=tract_id,
                    line=dict(color=color_map.get(tract_id, "gray"), width=2),
                    marker=dict(color=color_map.get(tract_id, "gray"), size=8),
                ))
        
        # Layout settings
        quant_fig.update_layout(
            title="Quantity",
            xaxis_title="Runs",
            yaxis_title="Log2(Ratio)" if sub_row_means_xic else "Log2(Intensity)",
            template="vizro",
            height=500,
            xaxis=dict(domain=[0, 0.8]),
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.82,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                orientation="v",
                traceorder="normal"
            ),
        )
        
        quality_fig.update_layout(
            title="Quality",
            xaxis_title="Runs",
            yaxis_title="XIC Corr",
            template="vizro",
            height=500,
            xaxis=dict(domain=[0, 0.8]),
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.82,
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
    [Output('run-dropdown', 'options'),
     Output('stored-report', 'data')],     
    Input('precursor-dropdown', 'value'),
    State('stored-outdir', 'data')
)
def update_run_options(selected_precursor: str, outdir_path: str) -> tuple:
    if selected_precursor is None or not outdir_path:
        return [], None

    try:
        out_pub_dir = Path(outdir_path) / 'publication'
        data_path = out_pub_dir / 'xic_quantification.parquet'
        required_cols = ['File.Name', 'Run', 'Precursor.Id', 'Precursor.Mz', 
                         'RT.Apex', 'RT.Left', 'RT.Right', 'Optimal.XIC', 'Optimal.XIC.Type', 
                         'n.MS1.XICs', 'n.MS2.XICs', 'MS1.XIC.Info', 'MS2.XIC.Info', 
                         'MS2.XIC.Corr.Raw', 'MS2.XIC.Quant.Raw', 'MS2.XIC.Quant.Adjusted',
                         'MS1.XIC.Corr.Raw', 'MS1.XIC.Quant.Raw', 'MS1.XIC.Quant.Adjusted']
        #selected_precursor = selected_precursor.split("_")[0]
        df = pd.read_parquet(data_path, 
                             columns=required_cols,
                             filters=[('Precursor.Id', '==', selected_precursor)])
        
        if df.empty:
            return [], None
        
        run_options = [{'label': run, 'value': run} for run in df['Run'].unique()]
        return run_options, df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error updating run options: {e}")
        return [], None

@app.callback(
    [Output('show-precid-i', 'children'), 
     Output('show-run-info', 'children'), 
     Output('xic-plot-ms1', 'figure'),
     Output('xic-plot-ms1-c', 'figure'),
     Output('xic-plot-ms2', 'figure'),
     Output('xic-plot-ms2-c', 'figure'),
     Output('xic-info-table', 'data')],
    Input('precursor-dropdown', 'value'),
    Input('show-ms1-inter', 'value'),
    Input('run-dropdown', 'value'),
    State('stored-report', 'data'),
    State('stored-outdir', 'data')
)
def update_XIC_plots_table(selected_precursor: str, 
                           show_interpolated_ms1: bool,
                           selected_run: str, 
                           stored_report: List[Dict[str, Any]], 
                           outdir_path: str) -> tuple:
    if selected_run is None or selected_precursor is None or not stored_report:
        return ('Select a run to check precursor XICs', '', go.Figure(), 
                go.Figure(), go.Figure(), go.Figure(), [])

    try:
        report_df = pd.DataFrame(stored_report)
        report_data = report_df[report_df['Run'] == selected_run].reset_index(drop=True)

        if report_data.empty:
            raise ValueError("No data found for selected run")
        
        # Get XIC file path using absolute paths
        out_int_dir = Path(outdir_path) / 'intermediate'
        xic_file_path = get_xic_path(out_int_dir, selected_run)
        if not xic_file_path.exists():
            raise FileNotFoundError(f"XIC file not found: {xic_file_path}")
        
        #selected_precursor = selected_precursor.split("_")[0]
        xic_data = pd.read_parquet(xic_file_path, filters=[('Precursor.Id', '==', selected_precursor)])
        
        if xic_data.empty:
            raise ValueError("No XIC data found for selected precursor")
    
        # Extract relevant data
        prec_title = f"     {selected_precursor}"
        run_title = f"     Run: {selected_run}"
        
        # Define template based on switch state
        template = "vizro" #if switch_on else "vizro_dark"
        
        # Precursor info table
        prec_info = report_data[['n.MS1.XICs',
                                 'MS1.XIC.Info', 
                                 'MS1.XIC.Corr.Raw', 
                                 'MS1.XIC.Quant.Raw', 
                                 'MS1.XIC.Quant.Adjusted']].iloc[0]
        n_ms1_xics = int(prec_info['n.MS1.XICs'])
        ms1_xics = prec_info['MS1.XIC.Info'].split(';')
        ms1_xic_names = []
        ms1_xic_mzs = []
        for ms1_xic in ms1_xics:
            parts = ms1_xic.split('/')
            ms1_xic_names.append(parts[0])
            ms1_xic_mzs.append(parts[1])

        prec_info_df = pd.DataFrame({
            "XIC Type": ["MS1"] * len(ms1_xics),
            "XIC Name": ms1_xic_names,
            "XIC m/z": ms1_xic_mzs,
            "XIC Corr": string2float(prec_info['MS1.XIC.Corr.Raw'].split(';'), 2),
            "XIC Quant (Raw)": string2float(prec_info['MS1.XIC.Quant.Raw'].split(';'), 2),
            "XIC Quant (Adjusted)": string2float(prec_info['MS1.XIC.Quant.Adjusted'].split(';'), 2)
        })

        # Fragment info table
        frag_info = report_data[['n.MS2.XICs',
                                 'MS2.XIC.Info', 
                                 'MS2.XIC.Corr.Raw', 
                                 'MS2.XIC.Quant.Raw', 
                                 'MS2.XIC.Quant.Adjusted']].iloc[0]
        n_ms2_xics = int(frag_info['n.MS2.XICs'])
        ms2_xics = frag_info['MS2.XIC.Info'].split(';')
        ms2_xic_names = []
        ms2_xic_mzs = []
        for ms2_xic in ms2_xics:
            parts = ms2_xic.split('/')
            ms2_xic_names.append(parts[0])
            ms2_xic_mzs.append(parts[1])
        
        frag_info_df = pd.DataFrame({
            "XIC Type": ["MS2"] * len(ms2_xics),
            "XIC Name": ms2_xic_names,
            "XIC m/z": ms2_xic_mzs,
            "XIC Corr": string2float(frag_info['MS2.XIC.Corr.Raw'].split(';'), 2),
            "XIC Quant (Raw)": string2float(frag_info['MS2.XIC.Quant.Raw'].split(';'), 2),
            "XIC Quant (Adjusted)": string2float(frag_info['MS2.XIC.Quant.Adjusted'].split(';'), 2)
        })

        optimal_xic_type = report_data['Optimal.XIC.Type'].iloc[0]
        optimal_xic = report_data['Optimal.XIC'].iloc[0]
        # annotate best ion
        if optimal_xic_type == 'ms2':
            best_frag = frag_info_df.loc[optimal_xic, 'XIC Name']
            frag_info_df.loc[optimal_xic, 'XIC Name'] = best_frag + " (optimal)"
        elif optimal_xic_type == 'ms1':
            best_prec = prec_info_df.loc[optimal_xic, 'XIC Name']
            prec_info_df.loc[optimal_xic, 'XIC Name'] = best_prec + " (optimal)"
        
        xic_mask = xic_data['row_type'] == 'xic_row'
        score_mask = xic_data['row_type'] == 'score_row'
        
        ms1_traces = []
        ms1_rt = xic_data.loc[xic_mask, 'ms1_rt'].values
        ms1_intensity_cols = xic_data.filter(like='ms1_intensity').columns[0:n_ms1_xics]
        for i, pr in enumerate(ms1_intensity_cols):
            ms1_intensity = xic_data.loc[xic_mask, pr].values
            ms1_traces.append(go.Scatter(
                x=ms1_rt,
                y=ms1_intensity,
                mode='lines+markers',
                name=ms1_xics[i], 
                line=dict(color=ms1_colors[ms1_xics[i].split("/")[0]]),
                marker=dict(color=ms1_colors[ms1_xics[i].split("/")[0]])
            ))
        
        ms2_traces = []
        ms2_rt = xic_data.loc[xic_mask, 'ms2_rt'].values
        ms2_intensity_cols = xic_data.filter(like='ms2_intensity').columns[0:n_ms2_xics]
        for i, fr in enumerate(ms2_intensity_cols):
            ms2_intensity = xic_data.loc[xic_mask, fr].values
            ms2_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms2_intensity,
                mode='lines+markers',
                name=ms2_xics[i], 
                line=dict(color=ms2_colors[i]),
                marker=dict(color=ms2_colors[i])
            ))
        
        ms1_i_traces = []
        ms1_interpol_cols = xic_data.filter(like='ms1_interpol').columns[0:n_ms1_xics]
        for i, pr in enumerate(ms1_interpol_cols):
            ms1_i_intensity = xic_data.loc[xic_mask, pr].values
            ms1_i_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms1_i_intensity,
                mode='lines+markers',
                name=ms1_xics[i],
                line=dict(color=ms1_colors[ms1_xics[i].split("/")[0]]),
                marker=dict(color=ms1_colors[ms1_xics[i].split("/")[0]])
            ))
            
        xic_c_path = out_int_dir / "xic_files" / "xics.adjusted.parquet"
        xic_data_c = pd.read_parquet(xic_c_path, 
                                     filters=[('Run', '==', selected_run),
                                              ('Precursor.Id', '==', selected_precursor)])
        ms1_traces_c = []
        ms1_rt_c = xic_data_c.loc[:, 'ms2_rt'].values
        for i, pr in enumerate(ms1_intensity_cols):
            ms1_intensity = xic_data_c.loc[:, pr].values
            ms1_traces_c.append(go.Scatter(
                x=ms1_rt_c,
                y=ms1_intensity,
                mode='lines+markers',
                name=ms1_xics[i],
                line=dict(color=ms1_colors[ms1_xics[i].split("/")[0]]),
                marker=dict(color=ms1_colors[ms1_xics[i].split("/")[0]])
            ))  
        
        ms2_traces_c = []
        ms2_rt_c = xic_data_c.loc[:, 'ms2_rt'].values
        for i, fr in enumerate(ms2_intensity_cols):
            ms2_intensity = xic_data_c.loc[:, fr].values
            ms2_traces_c.append(go.Scatter(
                x=ms2_rt_c,
                y=ms2_intensity,
                mode='lines+markers',
                name=ms2_xics[i],
                line=dict(color=ms2_colors[i]),
                marker=dict(color=ms2_colors[i])
            )) 
        
        if optimal_xic_type == 'ms2':
            ms2_traces_c.append(go.Scatter(
                x=ms2_rt_c,
                y=xic_data_c.loc[:, 'elution'].values,
                mode='lines+markers',
                name='optimal',
                marker=dict(size=6, color='red'),
                line=dict(dash='dash', color='red'),  # Use 'dash', 'dot', 'dashdot', etc.
            ))
        elif optimal_xic_type == 'ms1':
            ms1_traces_c.append(go.Scatter(
                x=ms1_rt_c,
                y=xic_data_c.loc[:, 'elution'].values,
                mode='lines+markers',
                name='optimal',
                marker=dict(size=6, color='red'),
                line=dict(dash='dash', color='red'),  # Use 'dash', 'dot', 'dashdot', etc.
            ))
        
        if not show_interpolated_ms1:
            fig_ms1 = go.Figure(data=ms1_traces)
        else:
            fig_ms1 = go.Figure(data=ms1_i_traces)
        
        fig_ms1.update_layout(
            title="MS1-level XICs (Raw)",
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            legend_title="",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                itemclick="toggleothers",
                itemdoubleclick=False,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
            ),
            xaxis_range=[min(ms1_rt[0], ms2_rt[0]), max(ms1_rt[-1], ms2_rt[-1])],
            yaxis=dict(tickformat=".1e"),
            template=template,
            margin=dict(l=0, r=0, t=70, b=70) 
        )
    
        fig_ms2 = go.Figure(data=ms2_traces)
        fig_ms2.update_layout(
            title="MS2-level XICs (Raw)",
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            legend_title="",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                itemclick="toggleothers",
                itemdoubleclick=False,
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
            ),
            xaxis_range=[min(ms1_rt[0], ms2_rt[0]), max(ms1_rt[-1], ms2_rt[-1])],
            yaxis=dict(tickformat=".1e"),
            template=template,
            margin=dict(l=0, r=0, t=70, b=70) 
        )
        
        fig_ms1_c = go.Figure(data=ms1_traces_c)
        fig_ms1_c.update_layout(
            title="MS1-level XICs (Adjusted)",
            xaxis_title="Retention Time",
            yaxis_title=None,
            showlegend=False,
            xaxis_range=[min(ms1_rt_c[0], ms2_rt_c[0]), max(ms1_rt_c[-1], ms2_rt_c[-1])],
            yaxis=dict(tickformat=".1e"),
            template=template,
            margin=dict(l=0, r=0, t=70, b=70) 
        )
    
        fig_ms2_c = go.Figure(data=ms2_traces_c)
        fig_ms2_c.update_layout(
            title="MS2-level XICs (Adjusted)",
            xaxis_title="Retention Time",
            yaxis_title=None,
            showlegend=False,
            xaxis_range=[min(ms1_rt_c[0], ms2_rt_c[0]), max(ms1_rt_c[-1], ms2_rt_c[-1])],
            yaxis=dict(tickformat=".1e"),
            template=template,
            margin=dict(l=0, r=0, t=70, b=70) 
        )
   
        # Add vertical lines to MS2 plot
        rt_marks = ['Apex', 'Left', 'Right']
        rt_boundaries = ms2_rt_c[[len(ms1_rt_c)//2, 0, -1]]
        for i, rt in enumerate(rt_boundaries):
            fig_ms1.add_vline(x=rt, line=dict(color="gray", dash="dash"), annotation_text=rt_marks[i], annotation_position="top right")
            fig_ms2.add_vline(x=rt, line=dict(color="gray", dash="dash"), annotation_text=rt_marks[i], annotation_position="top right")

        xic_info_df = pd.concat([prec_info_df.iloc[0:n_ms1_xics, :], frag_info_df.iloc[0:n_ms2_xics, :]], axis=0)
        xic_score = xic_data.loc[score_mask, ms1_intensity_cols.union(ms2_intensity_cols)].values
        xic_info_df.insert(3, 'XIC Rank Score', np.round(xic_score[0], 2))          
    except Exception as e:
        logger.error(f"Error updating plot: {e}")
        return '', '', go.Figure(), go.Figure(), go.Figure(), go.Figure(), []
    
    return prec_title, run_title, fig_ms1, fig_ms1_c, fig_ms2, fig_ms2_c, xic_info_df.to_dict('records')

