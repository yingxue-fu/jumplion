# builtin
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
# external
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objs as go
import plotly.express as px
# local
from jumpdiaq.gui.app import app, page_sidebar_style, logger
from jumpdiaq.gui.utils import extract_unique_parts


load_figure_template(["vizro", "vizro_dark"])


sidebar = html.Div(
    [
        html.H3("Precursor Quantification"),
        html.Hr(),
        dbc.Nav(
            [   
                html.Label(id='protein-label', style={"font-weight": "bold"}),
                dbc.Select(id='precursor-dropdown', placeholder="Select a Precursor"),
                html.Div(id='show-ms-level', style={'marginTop': '10px', 'color': 'green'}),
                html.Br(),
                html.Label("For cross runs", style={"font-weight": "bold"}),
                dbc.Switch(label='Show summarization value', id='show-prec-summ', 
                           value=False, className="d-inline-block", 
                           style={'marginTop': '5px'}),
                html.Br(),
                html.Label("For individual run", style={"font-weight": "bold"}),
                dbc.Select(id='run-dropdown', placeholder="Select a Run"),
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
    dcc.Store(id='data-store'),  # Added for storing DataFrame
    dbc.Row([
        dbc.Col(sidebar, width=4),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Cross runs', children=[
                        html.Div([   
                            dcc.Graph(id="prec-quant-plot"),
                            dcc.Graph(id="prec-quality-plot"),
                            ], className="p-4")
                ], label_style=tab_label_style),
                dbc.Tab(label='Individual run', children=[
                    dbc.Row([
                        dbc.Col(html.Div(id='text-output', style={'fontSize': '18px'}), width=11)
                    ], style={'margin-top': '20px'}),
                    dbc.Row([
                        dbc.Col(html.Div(id='show-prec-run', style={'color': 'green'}), width=11)
                    ], style={'margin-top': '20px'}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='xic-plot-ms1'), width=7),
                        dbc.Col(dcc.Graph(id='xic-plot-ms1-c'), width=4)
                    ], style={'margin-top': '5px'}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='xic-plot-ms2'), width=7),
                        dbc.Col(dcc.Graph(id='xic-plot-ms2-c'), width=4)
                    ], style={'margin-top': '5px'}),
                    dbc.Row([
                        dbc.Col(dash_table.DataTable(
                            id='xic-info-table',
                            columns=[{'name': col, 'id': col} for col in ["XIC Type", 
                                                                          "XIC Name",
                                                                          "XIC m/z", 
                                                                          "XIC Rank Score",
                                                                          "XIC Corr", 
                                                                          "XIC Quant (Raw)", 
                                                                          "XIC Quant (Adj)"]],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '5px'},
                            style_header={'fontWeight': 'bold'},
                            page_size=10,
                        ), width=11, className="mx-auto"),
                    ], style={'margin-top': '15px'}),
                    ], label_style=tab_label_style)
                ], style=tabs_styles),
        ], width=8)
    ])
], className="dbc dbc-row-selectable")


def validate_parquet_file(filepath: Path, required_cols: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate parquet file and required columns"""
    try:
        if not filepath.is_file():
            return False, f"File not found: {filepath}"
            
        if filepath.suffix.lower() != '.parquet':
            return False, f"Invalid file type. Expected .parquet file, got: {filepath.suffix}"
            
        if not os.access(filepath, os.R_OK):
            return False, f"File is not readable: {filepath}"
            
        df = pd.read_parquet(filepath, columns=required_cols)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
            
        return True, None
        
    except Exception as e:
        return False, str(e)

def get_xic_path(data_path: Path, run_name: str) -> Path:
    """
    Construct absolute path for XIC file
    """
    try:
        data_dir = data_path#.parent
        xic_path = (data_dir / "xic_files" / f"{run_name}.xic.parquet").resolve()
        return xic_path
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
def update_label(protein_name):
    return f"For protein: {protein_name}"


@app.callback(
    Output('precursor-dropdown', 'options'),
    Input('stored-precursor-options', 'data')
)
def load_options(stored_options):
    if stored_options is None:
        return []
    
    precursor_options = [{'label': precursor.split("_")[0], 'value': precursor} 
                         for precursor in stored_options]
    
    return precursor_options


# Callback to read HDF file based on selected row
@app.callback(
    Output('show-ms-level', 'children'), 
    Output('stored-precquant', 'data'),
    Output('stored-precquality', 'data'),
    Output('stored-precMS1quant', 'data'),
    Output('stored-precMS1quality', 'data'),
    Output('stored-precMS2quant', 'data'),
    Output('stored-precMS2quality', 'data'),
    Input('precursor-dropdown', 'value'),
    State('stored-outdir', 'data'),
    #prevent_initial_call=True
)
def read_precs_data(selected_precursor_ms, outdir_path):
    
    if not selected_precursor_ms or outdir_path is None:
        return None, None, None, None, None, None, None
    # outdir_path = r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\test_20250523"
    # df = pd.read_csv(Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\test_20250514\Protein.Group_Quan_maxLFQ.csv"))
    # meta = pd.read_csv(Path(r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\sample_meta.csv"))

    # Assuming "file_name" column contains the name of the corresponding HDF file
    prec_quant_file = os.path.join(outdir_path, "Precursor.Id_quant_ms1_ms2_summ_table.parquet")
    prec_quality_file = os.path.join(outdir_path, "Precursor.Id_quality_ms1_ms2_summ_table.parquet")
    
    ms1_quant_file = os.path.join(outdir_path, "Precursor.Id_MS1.XIC.Quant.Adjusted_mapping.parquet")
    ms1_quality_file = os.path.join(outdir_path, "Precursor.Id_MS1.XIC.Corr.Raw_mapping.parquet")
    ms2_quant_file = os.path.join(outdir_path, "Precursor.Id_MS2.XIC.Quant.Adjusted_mapping.parquet")
    ms2_quality_file = os.path.join(outdir_path, "Precursor.Id_MS2.XIC.Corr.Raw_mapping.parquet")
    
    selected_precursor = selected_precursor_ms.split("_")[0]
    ms_level = selected_precursor_ms.split("_")[1]
    ms_level_info = f"{ms_level}-level quantification of this precursor is used when summarizing to protein."
    try:
        prec_quant_df = pd.read_parquet(prec_quant_file, filters=[('Precursor.Id', '==', selected_precursor)])
        prec_quality_df = pd.read_parquet(prec_quality_file, filters=[('Precursor.Id', '==', selected_precursor)])
        
        ms1_quant_df = pd.read_parquet(ms1_quant_file, filters=[('target_id', '==', selected_precursor)])
        ms1_quant_df['samples'] = extract_unique_parts(ms1_quant_df['samples'].values.tolist())
        
        ms1_quality_df = pd.read_parquet(ms1_quality_file, filters=[('target_id', '==', selected_precursor)])
        ms1_quality_df['samples'] = extract_unique_parts(ms1_quality_df['samples'].values.tolist())
        
        ms2_quant_df = pd.read_parquet(ms2_quant_file, filters=[('target_id', '==', selected_precursor)])
        ms2_quant_df['samples'] = extract_unique_parts(ms2_quant_df['samples'].values.tolist())
        
        ms2_quality_df = pd.read_parquet(ms2_quality_file, filters=[('target_id', '==', selected_precursor)])
        ms2_quality_df['samples'] = extract_unique_parts(ms2_quality_df['samples'].values.tolist())
        
        return (ms_level_info,
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


# Callback to generate figure from selected row
@app.callback(
    Output('prec-quant-plot', 'figure'),
    Output('prec-quality-plot', 'figure'),
    Input('show-prec-summ', 'value'),
    Input('stored-precquant', 'data'),
    Input('stored-precquality', 'data'),
    Input('stored-precMS1quant', 'data'),
    Input('stored-precMS1quality', 'data'),
    Input('stored-precMS2quant', 'data'),
    Input('stored-precMS2quality', 'data'), 
    State('stored-meta', 'data'),
    prevent_initial_call=True  # Add this to prevent initial callback
)
def update_summ_plot(show_summ_value, 
                     prec_quant_data, 
                     prec_quality_data,
                     ms1_quant_data, 
                     ms1_quality_data,
                     ms2_quant_data, 
                     ms2_quality_data,
                     meta):
    ctx = callback_context
    if not ctx.triggered or not prec_quant_data or not ms1_quant_data:
        return (go.Figure(layout=dict(title="Select a precursor to visualize its quantification data")),
                go.Figure(layout=dict(title="Select a precursor to visualize its quality data")))
    
    try:
        # AVAVDLGNR2, Q8TCS8
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
        
        ms1_xic_names = ['M+0', 'M+1', 'M+2']
        def pivot_order_melt(df, value_type, samples, is_ms1):
            # Pivot to wide format
            df_pivot = df.pivot(index='source_id', columns='samples', values=f'source_{value_type}')
        
            # Add missing samples as NaN
            for sample in samples:
                if sample not in df_pivot.columns:
                    df_pivot[sample] = np.nan
        
            # Reorder columns
            df_pivot = df_pivot[samples]
        
            if is_ms1:
                # Get min mz for each source_id
                ms1_mzs = df['source_id'].unique()
                ms1_mzs = np.sort(ms1_mzs)
        
                # Assign labels based on mz rank
                label_map = dict(zip(ms1_mzs, ms1_xic_names[:len(ms1_mzs)]))
        
                # Rename index
                df_pivot = df_pivot.rename(index=label_map)
                df_pivot.index.name = 'source_id'
        
            # Convert to long format
            df_long = df_pivot.reset_index().melt(id_vars='source_id', var_name='X', value_name='Y')
            return df_long
        
        # Create figure
        quant_fig = go.Figure()
        quality_fig = go.Figure()
        
        # Plot each tract as a connected scatter plot
        meta_samples = extract_unique_parts(meta_samples)
        
        ms1_quant_df = pivot_order_melt(ms1_quant_df, 'quant', meta_samples, True)
        for tract_id, group in ms1_quant_df.groupby("source_id"):
            quant_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else None),
                marker=dict(color='gray' if show_summ_value else None),
                showlegend=not show_summ_value
            ))
        ms2_quant_df = pivot_order_melt(ms2_quant_df, 'quant', meta_samples, False)
        for tract_id, group in ms2_quant_df.groupby("source_id"):
            quant_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else None),
                marker=dict(color='gray' if show_summ_value else None),
                showlegend=not show_summ_value
            ))
        
        ms1_quality_df = pivot_order_melt(ms1_quality_df, 'quality', meta_samples, True)
        for tract_id, group in ms1_quality_df.groupby("source_id"):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else None),
                marker=dict(color='gray' if show_summ_value else None),
                showlegend=not show_summ_value
            ))
        ms2_quality_df = pivot_order_melt(ms2_quality_df, 'quality', meta_samples, False)
        for tract_id, group in ms2_quality_df.groupby("source_id"):
            quality_fig.add_trace(go.Scatter(
                x=group["X"], 
                y=group["Y"], 
                mode="lines+markers",
                name="Precursors" if show_summ_value else tract_id,
                line=dict(color='gray' if show_summ_value else None),
                marker=dict(color='gray' if show_summ_value else None),
                showlegend=not show_summ_value
            ))
        
        color_map = {"MS1": "purple", "MS2": "blue"}
        if show_summ_value:
            for tract_id, group in prec_quant_values.groupby("MS_type"):
                quant_fig.add_trace(go.Scatter(
                    x=group["X"], 
                    y=group["Y"], 
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
            #title=pg_name,
            xaxis_title="Runs",
            yaxis_title="log2(intensity)",
            template="vizro",
            height=500,
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
            #title=pg_name,
            xaxis_title="Runs",
            yaxis_title="Quality",
            template="vizro",
            height=500,
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
    [Output('run-dropdown', 'options'),
     Output('data-store', 'data')],     
     #Output('corr-table', 'data'),
     #Output('quan-table', 'data')],
    Input('precursor-dropdown', 'value'),
    State('stored-outdir', 'data')
)
def update_run_options(selected_precursor: str, file_path: str) -> tuple:
    if selected_precursor is None or not file_path:
        return [], None#, [], []

    try:
        # file_path = r'C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\test_20250513'
        data_path = Path(file_path) / 'xic_quan_report.parquet'
        required_cols = ['File.Name', 'Run', 'Precursor.Id', 'Precursor.Mz', 'MS1.XIC.Mzs', 
                            'RT.Apex', 'RT.Left', 'RT.Right', 'MS2.XIC.Info', 'MS2.XIC.Mzs', 
                            'Optimal.XIC', 'Optimal.XIC.Type',
                            'MS2.XIC.Corr.Raw', 'MS2.XIC.Quant.Raw', 'MS2.XIC.Quant.Adjusted',
                            'MS1.XIC.Corr.Raw', 'MS1.XIC.Quant.Raw', 'MS1.XIC.Quant.Adjusted']
        selected_precursor = selected_precursor.split("_")[0]
        df = pd.read_parquet(data_path, 
                           columns=required_cols,
                           filters=[('Precursor.Id', '==', selected_precursor)])
        
        if df.empty:
            return [], None#, [], []
        
        # Store the absolute path
        df['file_path'] = str(data_path)
        
        run_options = [{'label': run, 'value': run} for run in df['Run'].unique()]
        return run_options, df.to_dict('records')#, cross_run_corr, cross_run_quan
        
    except Exception as e:
        logger.error(f"Error updating run options: {e}")
        return [], None#, [], []

@app.callback(
    [Output('text-output', 'children'), 
     Output('show-prec-run', 'children'), 
     Output('xic-plot-ms1', 'figure'),
     Output('xic-plot-ms1-c', 'figure'),
     Output('xic-plot-ms2', 'figure'),
     Output('xic-plot-ms2-c', 'figure'),
     Output('xic-info-table', 'data')],
    [Input('run-dropdown', 'value'),
     Input('precursor-dropdown', 'value'),
     Input('show-ms1-inter', 'value'),],
    [State('data-store', 'data'),
     State('stored-outdir', 'data')]
)
def update_XIC_plots_table(selected_run: str, 
                            selected_precursor: str, 
                            show_interpolated_ms1: bool,
                            stored_data: List[Dict[str, Any]], 
                            file_path: str) -> tuple:
    if selected_run is None or selected_precursor is None or not stored_data:
        return '', '', go.Figure(), go.Figure(), go.Figure(), go.Figure(), []

    try:
        report_df = pd.DataFrame(stored_data)
        report_data = report_df[report_df['Run'] == selected_run].reset_index(drop=True)

        if report_data.empty:
            raise ValueError("No data found for selected run")
        
        # Get XIC file path using absolute paths
        data_path = Path(file_path)
        xic_path = get_xic_path(data_path, selected_run)
        
        logger.info(f"Looking for XIC file at: {xic_path}")
        
        if not xic_path.exists():
            raise FileNotFoundError(f"XIC file not found: {xic_path}")
        
        # xic_path = r"C:\Users\yfu\Downloads\miss_dia_test_data\quant_ums\test_20250604\xic_files\202206012_K2E50_ZI_500ul60C3cm5min_P1-D5_1_9423.xic.parquet"
        # selected_precursor = 'GTPEQPQC(UniMod:4)GFSNAVVQILR3'
        selected_precursor = selected_precursor.split("_")[0]
        xic_data = pd.read_parquet(xic_path, filters=[('Precursor.Id', '==', selected_precursor)])
        
        if xic_data.empty:
            raise ValueError("No XIC data found for selected precursor")
    
        # Extract relevant data
        title = html.Div([f"     {selected_precursor}"])
        
        prec_run_info = f"     {selected_run}"
        
        # Define template based on switch state
        template = "vizro" #if switch_on else "vizro_dark"
        
        #print('ms1 plots')
        optimal_xic_type = report_data['Optimal.XIC.Type'].iloc[0]
        optimal_xic = report_data['Optimal.XIC'].iloc[0]
        
        xic_mask = xic_data['row_type'] == 'xic_row'
        mz_mask = xic_data['row_type'] == 'mz_row'
        score_mask = xic_data['row_type'] == 'score_row'
        
        ms1_traces = []
        ms1_rt = xic_data.loc[xic_mask, 'ms1_rt'].values
        ms1_intensity_cols = xic_data.filter(like='ms1_intensity').columns
        pr_mzs = xic_data.loc[mz_mask, ms1_intensity_cols].values[0]
        for i, pr in enumerate(ms1_intensity_cols):
            ms1_intensity = xic_data.loc[xic_mask, pr].values
            ms1_traces.append(go.Scatter(
                x=ms1_rt,
                y=ms1_intensity,
                mode='lines+markers',
                name=f"m/z {np.round(pr_mzs[i], 4)}",
                marker=dict(size=6),
                #legendgroup="ms1"
            ))
        
        ms2_traces = []
        ms2_rt = xic_data.loc[xic_mask, 'ms2_rt'].values
        ms2_intensity_cols = xic_data.filter(like='ms2_intensity').columns
        for fr in ms2_intensity_cols:
            fr_mz = xic_data.loc[mz_mask, fr].values[0]
            ms2_intensity = xic_data.loc[xic_mask, fr].values
            ms2_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms2_intensity,
                mode='lines+markers',
                name=f"m/z {np.round(fr_mz, 4)}",
                #opacity=0.5,
                marker=dict(size=6),
                #legendgroup="ms2"
            ))
        
        ms1_i_traces = []
        ms1_interpol_cols = xic_data.filter(like='ms1_interpol').columns
        for i, pr in enumerate(ms1_interpol_cols):
            ms1_i_intensity = xic_data.loc[xic_mask, pr].values
            ms1_i_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms1_i_intensity,
                mode='lines+markers',
                name=f"m/z {np.round(pr_mzs[i], 4)}",
                marker=dict(size=6),
                #legendgroup="ms1"
            ))
            
        xic_c_path = (data_path / "xic_files" / "xics.corrected.parquet").resolve()
        xic_data_c = pd.read_parquet(xic_c_path, 
                                     filters=[('Run', '==', selected_run),
                                              ('Precursor.Id', '==', selected_precursor)])
        ms1_traces_c = []
        ms1_rt_c = xic_data_c.loc[:, 'ms2_rt'].values
        for pr in ms1_intensity_cols:
            ms1_intensity = xic_data_c.loc[:, pr].values
            ms1_traces_c.append(go.Scatter(
                x=ms1_rt_c,
                y=ms1_intensity,
                mode='lines+markers',
                marker=dict(size=6),
                #legendgroup="ms1"
            ))  
        
        ms2_traces_c = []
        ms2_rt_c = xic_data_c.loc[:, 'ms2_rt'].values
        for fr in ms2_intensity_cols:
            fr_mz = xic_data_c.loc[0, fr]
            ms2_intensity = xic_data_c.loc[:, fr].values
            ms2_traces_c.append(go.Scatter(
                x=ms2_rt_c,
                y=ms2_intensity,
                mode='lines+markers',
                marker=dict(size=6),
                #legendgroup="ms2"
            )) 
        
        if optimal_xic_type == 'ms2':
            ms2_traces_c.append(go.Scatter(
                x=ms2_rt_c,
                y=xic_data_c.loc[:, 'elution'].values,
                mode='lines+markers',
                #name=f"Fr {np.round(fr_mz, 4)}",
                marker=dict(size=6),
                line=dict(dash='dash'),  # Use 'dash', 'dot', 'dashdot', etc.
                #legendgroup="ms2"
            ))
        elif optimal_xic_type == 'ms1':
            ms1_traces_c.append(go.Scatter(
                x=ms1_rt_c,
                y=xic_data_c.loc[:, 'elution'].values,
                mode='lines+markers',
                #name=f"Fr {np.round(fr_mz, 4)}",
                marker=dict(size=6),
                line=dict(dash='dash'),  # Use 'dash', 'dot', 'dashdot', etc.
                #legendgroup="ms1"
            ))
        
        if not show_interpolated_ms1:
            fig_ms1 = go.Figure(data=ms1_traces)
            num_colors = len(ms1_traces)
        else:
            fig_ms1 = go.Figure(data=ms1_i_traces)
            num_colors = len(ms1_i_traces)
        
        ms1_colors = px.colors.sample_colorscale(px.colors.sequential.Jet, [i / (num_colors - 1) for i in range(num_colors)])
        fig_ms1.update_layout(
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            legend_title="MS1 XICs",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                itemclick="toggleothers",
                itemdoubleclick=False
            ),
            xaxis_range=[min(ms1_rt[0], ms2_rt[0]), max(ms1_rt[-1], ms2_rt[-1])],
            yaxis=dict(tickformat=".1e"),
            colorway=ms1_colors,
            template=template
        )
    
        fig_ms2 = go.Figure(data=ms2_traces)
        num_colors = len(ms2_traces)
        ms2_colors = px.colors.sample_colorscale(px.colors.sequential.Rainbow, [i / (num_colors - 1) for i in range(num_colors)])
        fig_ms2.update_layout(
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            legend_title="MS2 XICs",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                itemclick="toggleothers",
                itemdoubleclick=False
            ),
            xaxis_range=[min(ms1_rt[0], ms2_rt[0]), max(ms1_rt[-1], ms2_rt[-1])],
            yaxis=dict(tickformat=".1e"),
            colorway=ms2_colors,
            template=template
        )
        
        fig_ms1_c = go.Figure(data=ms1_traces_c)
        fig_ms1_c.update_layout(
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            showlegend=False,
            xaxis_range=[min(ms1_rt_c[0], ms2_rt_c[0]), max(ms1_rt_c[-1], ms2_rt_c[-1])],
            yaxis=dict(tickformat=".1e"),
            colorway=ms1_colors,
            template=template
        )
    
        fig_ms2_c = go.Figure(data=ms2_traces_c)
        fig_ms2_c.update_layout(
            xaxis_title="Retention Time",
            yaxis_title="Intensity",
            showlegend=False,
            xaxis_range=[min(ms1_rt_c[0], ms2_rt_c[0]), max(ms1_rt_c[-1], ms2_rt_c[-1])],
            yaxis=dict(tickformat=".1e"),
            colorway=ms2_colors,
            template=template
        )
        
        #print('add RT boundries')    
        # Add vertical lines to MS2 plot
        rt_marks = ['Apex', 'Left', 'Right']
        rt_boundaries = ms2_rt_c[[len(ms1_rt_c)//2, 0, -1]]
        for i, rt in enumerate(rt_boundaries):
            fig_ms1.add_vline(x=rt, line=dict(color="gray", dash="dash"), annotation_text=rt_marks[i], annotation_position="top left")
            fig_ms2.add_vline(x=rt, line=dict(color="gray", dash="dash"), annotation_text=rt_marks[i], annotation_position="top left")
        
        # Precursor info table
        prec_info = report_data[['MS1.XIC.Mzs', 
                                 'MS1.XIC.Corr.Raw', 
                                 'MS1.XIC.Quant.Raw', 
                                 'MS1.XIC.Quant.Adjusted']].iloc[0]
        prec_mz = prec_info['MS1.XIC.Mzs'].split(';')

        prec_info_df = pd.DataFrame({
            "XIC Type": ["MS1"] * len(prec_mz),
            "XIC Name": ['M+0', 'M+1', 'M+2'],
            "XIC m/z": string2float(prec_mz, 4),
            "XIC Corr": string2float(prec_info['MS1.XIC.Corr.Raw'].split(';'), 2),
            "XIC Quant (Raw)": string2float(prec_info['MS1.XIC.Quant.Raw'].split(';'), 2),
            "XIC Quant (Adj)": string2float(prec_info['MS1.XIC.Quant.Adjusted'].split(';'), 2)
        })

        # Fragment info table
        frag_info = report_data[['MS2.XIC.Info', 
                                 'MS2.XIC.Mzs',
                                 'MS2.XIC.Corr.Raw', 
                                 'MS2.XIC.Quant.Raw', 
                                 'MS2.XIC.Quant.Adjusted']].iloc[0]
        frag_ions = frag_info['MS2.XIC.Info'].split(';')
        frag_mzs = frag_info['MS2.XIC.Mzs'].split(';')
        
        frag_info_df = pd.DataFrame({
            "XIC Type": ["MS2"] * len(frag_ions),
            "XIC Name": frag_ions,
            "XIC m/z": string2float(frag_mzs, 4),
            "XIC Corr": string2float(frag_info['MS2.XIC.Corr.Raw'].split(';'), 2),
            "XIC Quant (Raw)": string2float(frag_info['MS2.XIC.Quant.Raw'].split(';'), 2),
            "XIC Quant (Adj)": string2float(frag_info['MS2.XIC.Quant.Adjusted'].split(';'), 2)
        })
        
        # annotate best ion
        if optimal_xic_type == 'ms2':
            best_frag = frag_info_df.loc[optimal_xic, 'XIC Name']
            frag_info_df.loc[optimal_xic, 'XIC Name'] = best_frag + " (optimal)"
        elif optimal_xic_type == 'ms1':
            best_prec = prec_info_df.loc[optimal_xic, 'XIC Name']
            prec_info_df.loc[optimal_xic, 'XIC Name'] = best_prec + " (optimal)"

        xic_info_df = pd.concat([prec_info_df, frag_info_df], axis=0)
        xic_score = xic_data.loc[score_mask, ms1_intensity_cols.union(ms2_intensity_cols)].values
        xic_info_df.insert(3, 'XIC Rank Score', np.round(xic_score[0], 2))

                                             
    except Exception as e:
        logger.error(f"Error updating plot: {e}")
        return '', '', go.Figure(), go.Figure(), go.Figure(), go.Figure(), []
    
    return title, prec_run_info, fig_ms1, fig_ms1_c, fig_ms2, fig_ms2_c, xic_info_df.to_dict('records')

