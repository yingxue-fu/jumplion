# builtin
import os
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError
from functools import lru_cache
from typing import List, Dict, Any#, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
# external
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, no_update#, callback_context
from dash_bootstrap_templates import load_figure_template, ThemeSwitchAIO
# local
from jumplion.utils import extract_unique_parts, define_species
from jumplion.gui.utils import cosine_similarity_vectors, topk_cosine_similarity
from jumplion.gui.app import (app, to_image_config, page_sidebar_style, logger,
                              THEME_LIGHT, THEME_DARK,
                              PLOTLY_TEMPLATE_LIGHT, PLOTLY_TEMPLATE_DARK)


load_figure_template(["vizro", "vizro_dark"])
ms1_colors = dict(zip([f'M+{i}' for i in range(9)], px.colors.qualitative.Set2[0:9]))
ms2_colors = px.colors.qualitative.Dark24_r
jumplion_level_colors = {'MS1': 'purple', 'MS2': 'blue', 'MS1_MS2': 'green'}
DIANN_PREC_FADED_COLOR = '#d3d3d3'  # light gray for background precursor traces
DIANN_PROT_COLOR = '#555555'        # darker gray for the DIA-NN protein line


XIC_INFO_TABLE_COLUMNS = [{'name': col, 'id': col} for col in [
    "Type", "Name", "m/z", "Correlation Score", "Shape Score", "Intensity Score",
    "Combined Score", "Reference Corr", "XIC Quant", "XIC Quant (Adj)",
]]

GITHUB_URL = "https://github.com/yingxue-fu/jumplion"
try:
    APP_VERSION = pkg_version("jumplion")
except PackageNotFoundError:
    APP_VERSION = "dev"

THEME_SWITCH_AIO_ID = "theme"
theme_switch = ThemeSwitchAIO(
    aio_id=THEME_SWITCH_AIO_ID,
    themes=[THEME_LIGHT, THEME_DARK],
)

TOPBAR_HEIGHT = "56px"

# The "XICs - All Runs" tab's precursor-id / legend / group-name info is all
# combined into ONE sticky row (per group, so the group name updates as you
# scroll between groups) rather than three stacked rows, so it only ever
# occupies a single row's worth of vertical space while scrolling. This is
# just below where the sticky nav-tabs bar ends (see assets/style.css).
ALL_RUNS_HEADER_STICKY_TOP = "105px"

topbar = dbc.Navbar(
    dbc.Container(
        [
            html.Div([
                html.Img(src="/assets/logo.png", height="32px", style={"marginRight": "10px"}),
                html.Span("JUMPlion Viewer", style={"fontWeight": "bold", "fontSize": "20px"}),
                dbc.Badge(f"v{APP_VERSION}", color="secondary", pill=True,
                          className="ms-2", style={"verticalAlign": "middle"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Div(theme_switch, style={"display": "none"}),  # hidden for now; callbacks still need it
                html.A(
                    html.I(className="fa-brands fa-github", style={"fontSize": "22px"}),
                    href=GITHUB_URL, target="_blank", rel="noopener noreferrer",
                    style={"marginLeft": "20px", "color": "inherit"},
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "15px"}),
        ],
        fluid=True,
        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
               "height": TOPBAR_HEIGHT, "paddingLeft": "2rem", "paddingRight": "2rem"},
    ),
    className="bg-body-tertiary",
    style={"position": "sticky", "top": 0, "zIndex": 1030,
           "borderBottom": "1px solid var(--bs-border-color)", "padding": 0},
)

sidebar = html.Div(
    [
        html.Div([
            html.H3("Inputs", style={"font-weight": "bold", "marginBottom": "0"}),
            html.Hr(style={"marginTop": "0.75rem", "marginBottom": "0.75rem"}),
        ], style={"position": "sticky", "top": "0", "zIndex": 10,
                  "backgroundColor": "var(--bs-body-bg, #fff)",
                  "marginTop": "1rem", "paddingTop": "0.75rem"}),
        dbc.Nav(
            [
                html.Label("Results directory", style={"font-weight": "bold"}),
                dbc.Input(id='outdir-path', type='text', placeholder='Path of output directory',
                          style={'width': '100%'}, persistence=True, debounce=True),
                html.Div(id='outdir-status', style={'fontSize': '13px', 'marginTop': '5px'}),
                html.Label("Meta file path", style={"font-weight": "bold", "margin-top": "10px"}),
                dbc.Input(id='meta-file-path', type='text', placeholder='Path of run meta info', 
                          style={'width': '100%'}, persistence=True),
                html.Div([
                    dbc.Button('Load precursors list',
                               id='load-prec-list', n_clicks=0, color='info',
                               style={"textTransform": "none"}),
                    dcc.Loading(
                        html.Div(id='load-prec-list-spinner'),
                        type='default', color='lightblue',
                        parent_style={'marginLeft': '20px', 'width': '30px', 'height': '30px'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}),
                html.Div(id='load-prec-list-status', style={'fontSize': '13px', 'marginTop': '5px'}),
                html.Hr(),
                html.Label('Select a precursor', style={"font-weight": "bold"}),
                dcc.Dropdown(id='precursor-dropdown', placeholder="Select a Precursor"),
                html.Label(id='protein-label', style={'marginTop': '5px'}),
                html.Br(),
                html.Label("Precursor summarization", style={"font-weight": "bold"}),
                dbc.Switch(label='Show relative ratio of quantity', id='sub-row-means-xic', value=False),
                dbc.Switch(label='Show summarized quantity', id='show-prec-summ', value=False),
                html.Div([
                    dbc.Checkbox(label='MS1', id='show-summ-ms1', value=False),
                    dbc.Checkbox(label='MS2', id='show-summ-ms2', value=False),
                ], style={'display': 'flex', 'gap': '1.5rem', 'marginLeft': '2rem'}),
                html.Br(),
                html.Label("Show precursor XICs across all runs", style={"font-weight": "bold"}),
                dbc.RadioItems(
                    id='xic-all-level',
                    options=[{'label': 'MS1', 'value': 'ms1'}, {'label': 'MS2', 'value': 'ms2'}],
                    value='ms1',
                    inline=True,
                    style={'marginTop': '5px'},
                ),
                dbc.Input(id='xic-all-radius', type='number', min=1, step=1,
                          placeholder='Radius around apex (# scan points, blank = full)',
                          style={'width': '100%', 'marginTop': '5px'}, persistence=True,
                          debounce=True),
                html.Br(),
                html.Label("Show precursor XICs in a run", style={"font-weight": "bold"}),
                dcc.Dropdown(id='run-dropdown', placeholder="Select a Run"),
                dbc.Switch(label='Show interpolated MS1 XICs', id='show-ms1-inter', value=False, 
                           className="d-inline-block", style={'marginTop': '5px'}),
                dbc.Switch(label='Show reference XIC', id='show-reference', value=False, 
                           className="d-inline-block"),
                dbc.Switch(label='Show isotope/fragment cosine similarity', id='show-cosine', value=False,
                           className="d-inline-block"),
                html.Hr(),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={**page_sidebar_style, "left": "0", "top": TOPBAR_HEIGHT, "paddingTop": "0rem",
           "backgroundColor": "var(--bs-body-bg, #fff)"},
)

tabs_styles = {
    'margin-top': '15px',
    #'padding-top': '15px'
}

tab_label_style = {
    'fontSize': '14px',
    "textTransform": "none",  # Prevent all caps
    "textAlign": "center"
}

layout = html.Div(children=[
    topbar,
    dcc.Store(id='stored-outdir', storage_type='session'),
    dcc.Store(id='stored-meta', storage_type='session'),
    dcc.Store(id='stored-precursor-options', storage_type='session'),
    dcc.Store(id='jumplion-prec-quant'),
    dcc.Store(id='diann-prec-quant'),
    dcc.Store(id='stored-prec-prot'),
    dcc.Store(id='prec-jumplion-quant'),  #
    dcc.Store(id='prec-diann-quant'),
    dcc.Store(id='ms1-xic-quants'),
    dcc.Store(id='ms2-xic-quants'),
    dcc.Store(id='stored-prec-report'),  # Added for storing DataFrame
    sidebar,
    html.Div([
            dbc.Tabs([
                dbc.Tab(label='Protein Quantification', tab_id='tab-prot-quant', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select a protein group", style={"font-weight": "bold"}),
                            dcc.Dropdown(id='prot-group-dropdown', placeholder="Select a Protein.Group"),
                        ], width=6),
                        dbc.Col([
                            dbc.Switch(label='Show relative ratio of quantity',
                                       id='prot-sub-row-means', value=False),
                            dbc.Switch(label='Show normalized precursor quantities',
                                       id='prot-use-norm', value=False),
                            dbc.Switch(label='Show protein quantity on the plots',
                                       id='show-diann-prot', value=False),
                        ], width=6),
                    ], style={'margin-top': '20px'}),
                    dcc.Loading(
                        html.Div(
                            html.Div("Select a protein group to visualize the quantification of its precursors."),
                            id='prot-quant-container', className="p-2"
                        ),
                        type='cube', color='lightblue'),
                ], label_style=tab_label_style),
                dbc.Tab(label='Precursor Quantification', tab_id='tab-quant', children=[
                        dcc.Loading([
                            dbc.Row([
                                dbc.Col(html.Div(id='show-precid-c', style={'fontSize': '20px'}), width=11)
                            ], style={'margin-top': '20px'}),
                            html.Div(
                                html.Div("Select a precursor to visualize its quantification results."),
                                id='quant-plots-container', className="p-2"
                            ),
                        ], type='cube', color='lightblue'),
                ], label_style=tab_label_style),
                dbc.Tab(label='XICs - All Runs', tab_id='tab-all-runs', children=[
                    dcc.Loading([
                        # Nested here (rather than the top-level store list) so
                        # dcc.Loading also covers the expensive raw-data-loading
                        # stage, not just the cheap redraw that follows it.
                        dcc.Store(id='xic-all-raw-store'),
                        html.Div(
                            html.Div("Select a precursor to visualize its XICs across all runs."),
                            id='xic-all-runs-container', className="p-2"
                        ),
                    ], type='cube', color='lightblue'),
                ], label_style=tab_label_style),
                dbc.Tab(label='XICs - Individual Run', tab_id='tab-individual', children=[
                    dbc.Row([
                        dbc.Col(html.Div(id='show-precid-i', style={'fontSize': '20px'}), width=11)
                    ], style={'margin-top': '20px'}),
                    dbc.Row([
                        dbc.Col(html.Div(id='show-run-info', style={'fontSize': '16px', 'color': 'green'}), width=11)
                    ], style={'margin-top': '5px'}),
                    html.Div(
                        html.Div("Select a precursor and run to view XICs."),
                        id='xic-individual-container', className="p-2"
                    ),
                    ], label_style=tab_label_style)
                ], id='main-tabs', active_tab='tab-quant', style=tabs_styles),
    ], style={'marginLeft': f"calc({page_sidebar_style['width']} + 1.5rem)", 'paddingRight': '1rem'}),
], className="dbc dbc-row-selectable")


# Final.Status (xic_quantification.parquet) -> (marker color, marker symbol)
FINAL_STATUS_STYLE = {
    'observed': ('#1f77b4', 'circle'),
    'recovered': ('#ff7f0e', 'diamond'),
    'unresolved': ('#d62728', 'x'),
}
# Per-run status columns stored next to the run quantities in 'prec-diann-quant'
FINAL_STATUS_PREFIX = 'Status::'


def diann_reported_status(row, run_label):
    """'reported' if DIA-NN observed this precursor in the run (Final.Status ==
    'observed'), else 'missing'; None when the row has no status for the run."""
    if row is None or f'{FINAL_STATUS_PREFIX}{run_label}' not in row.index:
        return None
    return 'reported' if row[f'{FINAL_STATUS_PREFIX}{run_label}'] == 'observed' else 'missing'


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


def parse_floats(value_strings: List[str]) -> List[float]:
    """Safely convert string numbers to floats (0.0 on failure)."""
    values = []
    for value in value_strings:
        try:
            values.append(float(value))
        except (ValueError, TypeError):
            values.append(0.0)
    return values


def normalize_to_pct(values: List[float]) -> List[float]:
    """Scale values to 0-100 as a percent of their own max (0 if all non-positive)."""
    max_val = max(values) if values else 0
    if max_val <= 0:
        return [0.0] * len(values)
    return [v / max_val * 100 for v in values]


@app.callback(
    Output('stored-outdir', 'data'),
    Output('outdir-status', 'children'),
    Input('outdir-path', 'value'),
)
def set_outdir(outdir_path):
    if not outdir_path:
        return None, ""
    if not os.path.isdir(outdir_path):
        return None, html.Span("Directory not found", style={'color': '#d62728'})
    return outdir_path, ""


# Callback to read the selected CSV file
@app.callback(
    [Output('jumplion-prec-quant', 'data'),
     Output('diann-prec-quant', 'data'),
     Output('stored-meta', 'data'),
     Output('stored-prec-prot', 'data'),
     Output('stored-precursor-options', 'data'),
     Output('load-prec-list-status', 'children'),
     Output('load-prec-list-spinner', 'children')],
    Input('load-prec-list', 'n_clicks'),
    State('stored-outdir', 'data'),
    State('meta-file-path', 'value'),
    prevent_initial_call=True
)
def load_prec_quant(n_clicks, outdir_path, meta_file_path):
    if not outdir_path or not meta_file_path:
        status = html.Span("Missing output directory or meta file path", style={'color': '#d62728'})
        return None, None, None, {}, [], status, ""

    out_int_dir = Path(outdir_path) / 'intermediate'
    out_pub_dir = Path(outdir_path) / 'publication'
    jumplion_prec_quant_path = out_int_dir / 'prec_quant' / 'precursor_quantity_all_levels.tsv'
    diann_prec_quant_path = out_pub_dir / 'prec_matrix_diann.tsv'
    prec_prot_relation_path = out_int_dir / "prec_prot_relation.tsv"

    try:
        # jumplion (one row per Precursor.Id per MS.Level: MS1, MS2, MS1_MS2)
        jumplion_quant_df = pd.read_csv(jumplion_prec_quant_path, sep='\t')
        # diann
        diann_quant_df = pd.read_csv(diann_prec_quant_path, sep='\t')
        # meta
        meta_df = pd.read_csv(meta_file_path)
        #
        precursor_options = jumplion_quant_df['Precursor.Id'].drop_duplicates().values.tolist()
        print(f"Number of precursors re-quantified by JUMPlion: {len(precursor_options)}")

        # Make the runs the same order as in the meta file
        run_names = meta_df['Run'].values.tolist()
        jumplion_quant_df = jumplion_quant_df.loc[:, ['Precursor.Id'] + run_names + ['MS.Level']]
        diann_quant_df = diann_quant_df.loc[:, ['Precursor.Id'] + run_names]

        # Simplify the run name
        simple_run_names = extract_unique_parts(run_names)
        jumplion_quant_df.columns = ['Precursor.Id'] + simple_run_names + ['MS.Level']
        diann_quant_df.columns = ['Precursor.Id'] + simple_run_names
        meta_df['Run_simp'] = simple_run_names

        # Combine with protein coverage info
        prec_prot_df = pd.read_csv(prec_prot_relation_path, sep='\t')
        prec_prot_df['Species'] = prec_prot_df['Protein.Names'].apply(define_species)

        status = html.Span(f"Loaded {len(precursor_options)} precursors", style={'color': '#2ca02c'})
        return (jumplion_quant_df.to_dict('records'),
                diann_quant_df.to_dict('records'),
                meta_df.to_dict('records'),
                prec_prot_df.to_dict('records'),
                precursor_options,
                status,
                "")

    except Exception as e:
        logger.error(f"Error loading quantification table: {str(e)}")
        status = html.Span(f"Error loading precursor list: {str(e)}", style={'color': '#d62728'})
        return None, None, None, {}, [], status, ""


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


@lru_cache(maxsize=4)
def _read_matrix_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')


def read_matrix(path: Path) -> pd.DataFrame:
    """Read a precursor matrix once per file version (re-read if the file changes)."""
    return _read_matrix_cached(str(path), path.stat().st_mtime)


@app.callback(
    Output('prot-group-dropdown', 'options'),
    Input('stored-meta', 'data'),
    State('stored-outdir', 'data'),
)
def load_prot_group_options(stored_meta, outdir_path):
    if not stored_meta or not outdir_path:
        return []
    try:
        final_df = read_matrix(Path(outdir_path) / 'publication' / 'prec_matrix_diann_jumplion.tsv')
        groups = sorted(final_df['Protein.Group'].dropna().unique().tolist())
        return [{'label': g, 'value': g} for g in groups]
    except Exception as e:
        logger.error(f"Error loading protein group list: {e}")
        return []


@app.callback(
    Output('prot-quant-container', 'children'),
    Input('prot-group-dropdown', 'value'),
    Input('prot-sub-row-means', 'value'),
    Input('prot-use-norm', 'value'),
    Input('show-diann-prot', 'value'),
    Input(ThemeSwitchAIO.ids.switch(THEME_SWITCH_AIO_ID), 'value'),
    State('stored-meta', 'data'),
    State('stored-outdir', 'data'),
)
def update_prot_quant_plots(protein_group, sub_row_means_xic, use_norm, show_diann_prot, light_theme,
                            stored_meta, outdir_path):
    """Two plots of all precursors of a Protein.Group across runs: DIA-NN matrix (left)
    and DIA-NN + JUMPlion recovered matrix (right)."""
    template_name = PLOTLY_TEMPLATE_LIGHT if light_theme else PLOTLY_TEMPLATE_DARK
    if not protein_group or not stored_meta or not outdir_path:
        return [html.Div("Select a protein group to visualize the quantification of its precursors.")]

    try:
        meta_df = pd.DataFrame(stored_meta)
        run_names = meta_df['Run'].values.tolist()
        simple_run_names = meta_df['Run_simp'].values.tolist()
        out_pub_dir = Path(outdir_path) / 'publication'

        suffix = '_norm' if use_norm else ''
        norm_label = " (Normalized)" if use_norm else ""
        sources = [
            (f"DIA-NN Precursor Quantification{norm_label}", f'prec_matrix_diann{suffix}.tsv'),
            (f"DIA-NN + JUMPlion Precursor Quantification{norm_label}", f'prec_matrix_diann_jumplion{suffix}.tsv'),
        ]
        group_dfs = []
        for _, file_name in sources:
            mat = read_matrix(out_pub_dir / file_name)
            if 'Protein.Group' not in mat.columns:
                # normalized table without protein info: take it from the raw table
                raw = read_matrix(out_pub_dir / file_name.replace('_norm', ''))
                mat = mat.merge(raw[['Precursor.Id', 'Protein.Group']], on='Precursor.Id', how='left')
            group_dfs.append(mat.loc[mat['Protein.Group'] == protein_group].reset_index(drop=True))

        precursor_ids = sorted(set().union(*[set(g['Precursor.Id']) for g in group_dfs]))
        if not precursor_ids:
            return [html.Div(f"No precursors found for {protein_group}.")]
        palette = px.colors.qualitative.Dark24
        color_map = {pid: palette[i % len(palette)] for i, pid in enumerate(precursor_ids)}

        # log2 values (non-positive/missing -> NaN), optionally relative to the precursor mean.
        # Both matrices are aligned to the same precursor list, so the plots share precursors.
        log2_values = []
        for g in group_dfs:
            aligned = g.set_index('Precursor.Id').reindex(precursor_ids)
            vals = np.log2(aligned[run_names].apply(pd.to_numeric, errors='coerce'))
            vals = vals.replace([np.inf, -np.inf], np.nan)
            if sub_row_means_xic:
                vals = vals.sub(vals.mean(axis=1), axis=0)
            log2_values.append(vals)

        # Optional protein-group quantity per panel: prot_matrix_diann.tsv (left) and
        # prot_matrix_diann_jumplion.tsv (right)
        prot_sources = [
            ('prot_matrix_diann.tsv', "DIA-NN protein"),
            ('prot_matrix_diann_jumplion.tsv', "DIA-NN + JUMPlion protein"),
        ]
        prot_log2_list = [None, None]
        prot_notes = []
        if show_diann_prot:
            for i, (file_name, _) in enumerate(prot_sources):
                prot_path = out_pub_dir / file_name
                if not prot_path.exists():
                    prot_notes.append(f"{file_name} not found")
                    continue
                prot_mat = read_matrix(prot_path)
                prot_row = prot_mat.loc[prot_mat['Protein.Group'] == protein_group]
                if prot_row.empty:
                    prot_notes.append(f"{protein_group} not found in {file_name}")
                    continue
                p = np.log2(prot_row[run_names].iloc[0].apply(pd.to_numeric, errors='coerce'))
                p = p.replace([np.inf, -np.inf], np.nan)
                if sub_row_means_xic:
                    p = p - p.mean()
                prot_log2_list[i] = p

        all_log2 = [v.to_numpy().ravel() for v in log2_values]
        all_log2.extend(p.to_numpy() for p in prot_log2_list if p is not None)
        finite = np.concatenate(all_log2)
        finite = finite[np.isfinite(finite)]
        y_range = None
        if finite.size:
            pad = max((finite.max() - finite.min()) * 0.05, 0.5)
            y_range = [float(finite.min() - pad), float(finite.max() + pad)]

        # One figure with two panels so the precursor legend is shared and sits underneath
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=[title for title, _ in sources])
        # A panel showing its protein quantity gets light-gray precursor traces (as in
        # update_summ_plot) that leave the legend; the precursor legend comes from the
        # first panel that is not faded (none if both are).
        faded_panels = [p is not None for p in prot_log2_list]
        legend_col = next((c for c in (1, 2) if not faded_panels[c - 1]), None)
        opacity = 0.7 if sub_row_means_xic else 1.0
        for col, vals in enumerate(log2_values, start=1):
            faded = faded_panels[col - 1]
            for pid in precursor_ids:
                trace_color = DIANN_PREC_FADED_COLOR if faded else color_map[pid]
                fig.add_trace(go.Scatter(
                    x=simple_run_names,
                    y=vals.loc[pid].to_numpy(),
                    mode="lines+markers",
                    name=pid,
                    legendgroup=pid,
                    showlegend=(col == legend_col),
                    line=dict(color=trace_color),
                    marker=dict(color=trace_color),
                    opacity=opacity,
                ), row=1, col=col)
        for col, (p, (_, prot_name)) in enumerate(zip(prot_log2_list, prot_sources), start=1):
            if p is None:
                continue
            fig.add_trace(go.Scatter(
                x=simple_run_names,
                y=p.to_numpy(),
                mode="lines+markers",
                name=prot_name,
                line=dict(color=DIANN_PROT_COLOR, width=2),
                marker=dict(color=DIANN_PROT_COLOR, size=8),
            ), row=1, col=col)
        fig.update_xaxes(title_text="Runs")
        fig.update_yaxes(title_text="Log2(Ratio)" if sub_row_means_xic else "Log2(Intensity)",
                         range=y_range, col=1)
        fig.update_yaxes(range=y_range, col=2)
        fig.update_layout(
            template=template_name,
            height=650,
            margin=dict(b=60),
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                        itemclick="toggleothers", itemdoubleclick="toggle",
                        title_text=f"Precursors ({len(precursor_ids)})" if legend_col else None),
        )

        info = group_dfs[1] if not group_dfs[1].empty else group_dfs[0]
        header = f"Protein.Group: {protein_group}"
        extra = [str(info[c].iloc[0]) for c in ('Genes', 'Protein.Names')
                 if c in info.columns and not info.empty and pd.notna(info[c].iloc[0])]
        if extra:
            header += "  |  " + "  |  ".join(extra)
        # n.Precursors / n.Peptides of the protein group (DIA-NN + JUMPlion protein table)
        count_path = out_pub_dir / 'prot_matrix_diann_jumplion.tsv'
        if count_path.exists():
            count_mat = read_matrix(count_path)
            count_row = count_mat.loc[count_mat['Protein.Group'] == protein_group]
            if not count_row.empty:
                counts = [f"{c}: {int(count_row[c].iloc[0])}" for c in ('n.Precursors', 'n.Peptides')
                          if c in count_row.columns and pd.notna(count_row[c].iloc[0])]
                if counts:
                    header += "  |  " + "  |  ".join(counts)
        if prot_notes:
            header += "  |  Protein quantity not shown: " + "; ".join(prot_notes)

        return [
            html.Div(header, style={'fontSize': '16px', 'marginBottom': '10px'}),
            dcc.Graph(figure=fig, config=to_image_config),
        ]
    except Exception as e:
        logger.error(f"Error updating protein quantification plots: {e}")
        return [html.Div(f"Error updating plots: {e}")]


@app.callback(
    Output('protein-label', 'children'),
    Input('precursor-dropdown', 'value'),
    State('stored-prec-prot', 'data'),
    prevent_initial_call=True
)
def show_prec_info(selected_precursor, prec_prot_dict):
    if selected_precursor is not None:
        prec_prot_df = pd.DataFrame(prec_prot_dict)
        protein_name = prec_prot_df.loc[prec_prot_df['Precursor.Id'] == selected_precursor, 'Protein.Names'].values[0]
        return f"Protein: {protein_name}"
    else:
        return ""



@app.callback(
    [Output('show-precid-c', 'children'),
     Output('prec-jumplion-quant', 'data'),
     Output('prec-diann-quant', 'data'),
     Output('ms1-xic-quants', 'data'),
     Output('ms2-xic-quants', 'data')],
    Input('precursor-dropdown', 'value'),
    State('jumplion-prec-quant', 'data'),
    State('diann-prec-quant', 'data'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
    prevent_initial_call=True
)
def read_selected_prec_data(selected_precursor, jumplion_prec_quant, diann_prec_quant, outdir_path, stored_meta):
    if not selected_precursor or outdir_path is None:
        return None, None, None, None, None
    
    out_prec_dir = Path(outdir_path) / 'intermediate' / 'prec_quant'
    #
    ms1_exist = False
    ms1_quant_file_list = list(out_prec_dir.glob("Precursor.Id_MS1.XIC.Quan*_mapping.parquet"))
    if len(ms1_quant_file_list):
        ms1_exist = True
        ms1_quant_file = ms1_quant_file_list[0]
    #
    ms2_exist = False
    ms2_quant_file_list = list(out_prec_dir.glob("Precursor.Id_MS2.XIC.Quan*_mapping.parquet"))
    if len(ms2_quant_file_list):
        ms2_exist = True
        ms2_quant_file = ms2_quant_file_list[0]
    
    precursor_id = selected_precursor
    title = html.Div([f"     {precursor_id}"])
    try:
        jumplion_prec_quant_df = pd.DataFrame(jumplion_prec_quant)
        prec_jumplion_quant = jumplion_prec_quant_df.loc[jumplion_prec_quant_df['Precursor.Id'] == precursor_id, :]
        #print(prec_jumplion_quant)
        
        # Final.Quantity / Final.Status (DIA-NN observed values plus JUMPlion
        # recovered ones) replace the DIA-NN matrix. Stored as one wide row
        # (run -> quantity) plus a 'Status::<run>' column per run.
        meta_df = pd.DataFrame(stored_meta)
        run_map = dict(zip(meta_df['Run'], meta_df['Run_simp']))
        final_df = pd.read_parquet(Path(outdir_path) / 'publication' / 'xic_quantification.parquet',
                                   columns=['Run', 'Final.Quantity', 'Final.Status'],
                                   filters=[('Precursor.Id', '==', precursor_id)])
        final_row = {'Precursor.Id': precursor_id}
        for run_simp in meta_df['Run_simp']:
            final_row[run_simp] = np.nan
            final_row[f'{FINAL_STATUS_PREFIX}{run_simp}'] = None
        for run, qty, stat in zip(final_df['Run'], final_df['Final.Quantity'], final_df['Final.Status']):
            run_simp = run_map.get(run)
            if run_simp is not None:
                final_row[run_simp] = qty
                final_row[f'{FINAL_STATUS_PREFIX}{run_simp}'] = stat
        prec_diann_quant = pd.DataFrame([final_row])
        #print(prec_diann_quant)
        
        if ms1_exist:
            ms1_xic_quants = pd.read_parquet(ms1_quant_file, filters=[('target_id', '==', precursor_id)])
            ms1_xic_quants = ms1_xic_quants[ms1_xic_quants['source_quantity'] > 0]
            ms1_xic_quants['source_quantity'] = np.log2(ms1_xic_quants['source_quantity'])
            ms1_xic_quants['run'] = extract_unique_parts(ms1_xic_quants['run'].values.tolist())
            #print(ms1_xic_quants)
        else:
            ms1_xic_quants = pd.DataFrame()
        
        if ms2_exist:
            ms2_xic_quants = pd.read_parquet(ms2_quant_file, filters=[('target_id', '==', precursor_id)])
            ms2_xic_quants = ms2_xic_quants[ms2_xic_quants['source_quantity'] > 0]
            ms2_xic_quants['source_quantity'] = np.log2(ms2_xic_quants['source_quantity'])
            ms2_xic_quants['run'] = extract_unique_parts(ms2_xic_quants['run'].values.tolist())
            #print(ms2_xic_quants)
        else:
            ms2_xic_quants = pd.DataFrame()
                
        return (title, 
                prec_jumplion_quant.to_dict("records"), 
                prec_diann_quant.to_dict("records"), 
                ms1_xic_quants.to_dict("records"), 
                ms2_xic_quants.to_dict("records"), 
                )
    except Exception as e:
        logger.error(f"Error loading parquet file: {str(e)}")
        return None, None, None, None, None


@app.callback(
    Output('quant-plots-container', 'children'),
    Input('sub-row-means-xic', 'value'),
    Input('show-prec-summ', 'value'),
    Input('show-summ-ms1', 'value'),
    Input('show-summ-ms2', 'value'),
    Input('prec-jumplion-quant', 'data'),
    Input('prec-diann-quant', 'data'),
    Input('ms1-xic-quants', 'data'),
    Input('ms2-xic-quants', 'data'),
    Input(ThemeSwitchAIO.ids.switch(THEME_SWITCH_AIO_ID), 'value'),
    State('stored-meta', 'data'),
    State('stored-outdir', 'data'),
    prevent_initial_call=True
)
def update_summ_plot(sub_row_means_xic,
                     show_summ_value,
                     show_summ_ms1,
                     show_summ_ms2,
                     prec_jumplion_quant,
                     prec_diann_quant,
                     ms1_xic_quants,
                     ms2_xic_quants,
                     light_theme,
                     meta,
                     outdir_path):
    template_name = PLOTLY_TEMPLATE_LIGHT if light_theme else PLOTLY_TEMPLATE_DARK
    if not prec_jumplion_quant or not prec_diann_quant:
        return [html.Div("Select a precursor to visualize its quantification results.")]

    try:
        meta = pd.DataFrame(meta)
        simple_run_names = meta['Run_simp'].values.tolist()
        # 
        prec_jumplion_quant = pd.DataFrame(prec_jumplion_quant)
        precursor_id = (prec_jumplion_quant['Precursor.Id'].iloc[0]
                         if 'Precursor.Id' in prec_jumplion_quant.columns and not prec_jumplion_quant.empty
                         else None)
        prec_jumplion_quant = prec_jumplion_quant.loc[:, simple_run_names + ['MS.Level']]
        prec_jumplion_quant_values = prec_jumplion_quant.melt(id_vars='MS.Level', var_name='X', value_name='Y')
        #
        prec_diann_quant = pd.DataFrame(prec_diann_quant)
        final_statuses = [prec_diann_quant[f'{FINAL_STATUS_PREFIX}{r}'].iloc[0]
                          if f'{FINAL_STATUS_PREFIX}{r}' in prec_diann_quant.columns else None
                          for r in simple_run_names]
        prec_diann_quant = prec_diann_quant.loc[:, simple_run_names]
        prec_diann_quant_values = prec_diann_quant.melt(var_name='X', value_name='Y')
        prec_diann_quant_values['Status'] = final_statuses  # one row, so melt keeps run order
        # Retrieve selected row data
        ms1_xic_quants = pd.DataFrame(ms1_xic_quants)
        ms2_xic_quants = pd.DataFrame(ms2_xic_quants)
        
        def pivot_order_melt(df, value_type, runs):
            # Pivot to wide format
            df_pivot = df.pivot(index='source_id', 
                                columns='run', 
                                values=f'source_{value_type}')
            # Add missing runs as NaN
            for run in runs:
                if run not in df_pivot.columns:
                    df_pivot[run] = np.nan
            # Reorder columns
            df_pivot = df_pivot[runs]
            # Convert to long format
            df_long = df_pivot.reset_index().melt(id_vars='source_id', 
                                                  var_name='X', 
                                                  value_name='Y')
            return df_long
        
        # Create figure
        jumplion_quant_fig = go.Figure()
        diann_quant_fig = go.Figure()
        jumplion_y_values = []
        diann_y_values = []

        # Plot each tract as a connected scatter plot
        if not ms1_xic_quants.empty:
            ms1_xic_quants = pivot_order_melt(ms1_xic_quants, 'quantity', simple_run_names)
            #print(ms1_xic_quants)
            for tract_id, group in ms1_xic_quants.groupby("source_id"):
                x=group["X"]
                if sub_row_means_xic:
                    y=group["Y"]-np.nanmean(group["Y"].to_numpy())
                    opacity=0.7
                else:
                    y=group["Y"]
                    opacity=1.0
                jumplion_quant_fig.add_trace(go.Scatter(
                    x=x, 
                    y=y, 
                    mode="lines+markers",
                    name="MS1 XICs" if show_summ_value else tract_id,
                    line=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                    marker=dict(color='#E7DDFF' if show_summ_value else ms1_colors[tract_id.split("/")[0]]),
                    showlegend=not show_summ_value,
                    opacity=opacity
                ))
                jumplion_y_values.append(y)

        if not ms2_xic_quants.empty:
            ms2_xic_quants = pivot_order_melt(ms2_xic_quants, 'quantity', simple_run_names)
            #print(ms2_xic_quants)
            for i, (tract_id, group) in enumerate(ms2_xic_quants.groupby("source_id")):
                x=group["X"]
                if sub_row_means_xic:
                    y=group["Y"]-np.nanmean(group["Y"].to_numpy())
                    opacity=0.7
                else:
                    y=group["Y"]
                    opacity=1.0
                jumplion_quant_fig.add_trace(go.Scatter(
                    x=x, 
                    y=y, 
                    mode="lines+markers",
                    name="MS2 XICs" if show_summ_value else tract_id,
                    line=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                    marker=dict(color='#8fd3fe' if show_summ_value else ms2_colors[i]),
                    showlegend=not show_summ_value,
                    opacity=opacity
                ))
                jumplion_y_values.append(y)

        if show_summ_value:
            # MS1_MS2 is always shown; MS1/MS2 only when their checkbox is ticked
            summ_levels = ['MS1_MS2']
            if show_summ_ms1:
                summ_levels.insert(0, 'MS1')
            if show_summ_ms2:
                summ_levels.insert(1 if show_summ_ms1 else 0, 'MS2')
            for level in summ_levels:
                level_values = prec_jumplion_quant_values.loc[prec_jumplion_quant_values['MS.Level'] == level]
                if level_values.empty:
                    continue
                x = level_values["X"]
                y_log2 = np.log2(pd.to_numeric(level_values["Y"], errors='coerce'))
                if sub_row_means_xic:
                    y = y_log2 - np.nanmean(y_log2.to_numpy())
                    opacity = 0.7
                else:
                    y = y_log2
                    opacity = 1.0
                jumplion_quant_fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=level,
                    line=dict(color=jumplion_level_colors.get(level, "blue"), width=2),
                    marker=dict(color=jumplion_level_colors.get(level, "blue"), size=8),
                ))
                jumplion_y_values.append(y)

        #
        x=prec_diann_quant_values["X"]
        y_log2=np.log2(pd.to_numeric(prec_diann_quant_values["Y"], errors='coerce'))
        if sub_row_means_xic:
            y=y_log2-np.nanmean(y_log2.to_numpy())
            opacity=0.7
        else:
            y=y_log2
            opacity=1.0
        # Connecting line, then one marker trace per Final.Status
        diann_quant_fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="#999999", width=1.5),
            showlegend=False,
            hoverinfo='skip',
        ))
        for status, (status_color, status_symbol) in FINAL_STATUS_STYLE.items():
            if status == 'unresolved':  # no quantity to draw, keep it out of the legend
                continue
            mask = (prec_diann_quant_values['Status'] == status).to_numpy()
            if not mask.any():
                continue
            diann_quant_fig.add_trace(go.Scatter(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=status,
                marker=dict(color=status_color, size=9, symbol=status_symbol),
            ))
        diann_y_values.append(y)

        # Match the two plots' y-axis span (ymax - ymin) while letting each
        # keep its own data-centered range.
        def _y_min_max(y_values_list):
            all_y = np.concatenate([np.asarray(y, dtype=float).ravel() for y in y_values_list]) \
                if y_values_list else np.array([])
            finite = all_y[np.isfinite(all_y)]
            if finite.size == 0:
                return None
            return float(np.min(finite)), float(np.max(finite))

        jumplion_minmax = _y_min_max(jumplion_y_values)
        diann_minmax = _y_min_max(diann_y_values)

        jumplion_yaxis_range = None
        diann_yaxis_range = None
        if jumplion_minmax and diann_minmax:
            j_min, j_max = jumplion_minmax
            d_min, d_max = diann_minmax
            max_span = max(j_max - j_min, d_max - d_min)
            if max_span <= 0:
                max_span = 1.0
            half = max_span / 2 * 1.1  # small padding
            jumplion_yaxis_range = [(j_min + j_max) / 2 - half, (j_min + j_max) / 2 + half]
            diann_yaxis_range = [(d_min + d_max) / 2 - half, (d_min + d_max) / 2 + half]

        # Layout settings
        jumplion_quant_fig.update_layout(
            title="JUMPlion Raw Quantification",
            xaxis_title="Runs",
            yaxis_title="Log2(Ratio)" if sub_row_means_xic else "Log2(Intensity)",
            template=template_name,
            height=500,
            xaxis=dict(domain=[0, 0.8]),
            yaxis=dict(range=jumplion_yaxis_range),
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

        diann_quant_fig.update_layout(
            title="Final Quantification (DIA-NN observed + JUMPlion recovered)",
            xaxis_title="Runs",
            yaxis_title="Log2(Intensity)",
            template=template_name,
            height=500,
            xaxis=dict(domain=[0, 0.8]),
            yaxis=dict(range=diann_yaxis_range),
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
        
        # Scatter comparing JUMPlion's combined MS1_MS2 quantification against
        # DIA-NN's, one point per run, for the currently selected precursor.
        ms1_ms2_values = prec_jumplion_quant_values.loc[
            prec_jumplion_quant_values['MS.Level'] == 'MS1_MS2', ['X', 'Y']
        ].rename(columns={'Y': 'jumplion'})
        diann_values = prec_diann_quant_values.loc[:, ['X', 'Y', 'Status']].rename(columns={'Y': 'diann'})
        merged = pd.merge(ms1_ms2_values, diann_values, on='X', how='inner')
        merged['jumplion_log2'] = np.log2(pd.to_numeric(merged['jumplion'], errors='coerce'))
        merged['diann_log2'] = np.log2(pd.to_numeric(merged['diann'], errors='coerce'))
        merged = merged.dropna(subset=['jumplion_log2', 'diann_log2'])

        # Per-precursor linear calibration model fit between the two
        # (log2(DIA-NN) = Intercept + Slope * log2(JUMPlion)); see
        # jumplion_precursor_linear_rescue_v2_fast.py.
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() == 'true'
            return bool(v)

        calib_row = None
        if outdir_path and precursor_id is not None:
            calib_path = (Path(outdir_path) / 'intermediate' / 'prec_quant' /
                          'precursor_mapping_model_metrics.tsv')
            try:
                if calib_path.exists():
                    calib_df = pd.read_csv(calib_path, sep='\t')
                    match = calib_df.loc[calib_df['Precursor.Id'] == precursor_id]
                    if not match.empty:
                        calib_row = match.iloc[0]
            except Exception as calib_err:
                logger.error(f"Error loading calibration model metrics: {calib_err}")

        def _nice_step(span, target_ticks=6):
            if span <= 0:
                return 1.0
            raw_step = span / target_ticks
            magnitude = 10 ** np.floor(np.log10(raw_step))
            residual = raw_step / magnitude
            if residual < 1.5:
                nice = 1
            elif residual < 3:
                nice = 2
            elif residual < 7:
                nice = 5
            else:
                nice = 10
            return nice * magnitude

        corr_fig = go.Figure()
        xaxis_range = None
        yaxis_range = None
        common_dtick = None
        if not merged.empty:
            x_min, x_max = merged['jumplion_log2'].min(), merged['jumplion_log2'].max()
            y_min, y_max = merged['diann_log2'].min(), merged['diann_log2'].max()

            # Give both axes the same (max - min) span, each centered on its
            # own data, so the plot area reads on a consistent visual scale.
            max_span = max(x_max - x_min, y_max - y_min)
            if max_span <= 0:
                max_span = 1.0
            half = max_span / 2 * 1.1  # small padding
            xaxis_range = [(x_min + x_max) / 2 - half, (x_min + x_max) / 2 + half]
            yaxis_range = [(y_min + y_max) / 2 - half, (y_min + y_max) / 2 + half]
            common_dtick = _nice_step(max_span * 1.1)

            has_model = (calib_row is not None
                         and pd.notna(calib_row.get('Slope'))
                         and pd.notna(calib_row.get('Intercept')))

            if has_model:
                slope = float(calib_row['Slope'])
                intercept = float(calib_row['Intercept'])
                line_x = xaxis_range
                line_y = [intercept + slope * xv for xv in xaxis_range]
                line_name = f"Model fit: y = {slope:.3f}x + {intercept:.3f}"
            else:
                lo, hi = min(xaxis_range[0], yaxis_range[0]), max(xaxis_range[1], yaxis_range[1])
                line_x = [lo, hi]
                line_y = [lo, hi]
                line_name = "y = x (no calibration model)"

            corr_fig.add_trace(go.Scatter(
                x=line_x, y=line_y, mode='lines',
                line=dict(color='gray', dash='dash'), name=line_name,
            ))
            for status, (status_color, status_symbol) in FINAL_STATUS_STYLE.items():
                sub = merged[merged['Status'] == status]
                if sub.empty:
                    continue
                corr_fig.add_trace(go.Scatter(
                    x=sub['jumplion_log2'],
                    y=sub['diann_log2'],
                    mode='markers',
                    text=sub['X'],
                    hovertemplate='%{text}<br>JUMPlion: %{x:.2f}<br>Final: %{y:.2f}<extra></extra>',
                    marker=dict(size=8, color=status_color, symbol=status_symbol),
                    name=status,
                ))

            if has_model:
                model_good = _to_bool(calib_row.get('Model.Good'))
                model_valid = _to_bool(calib_row.get('Model.Valid'))
                status = "Good" if model_good else ("Valid" if model_valid else "Not used for rescue")
                r2 = calib_row.get('R2')
                mae = calib_row.get('MAE')
                pearson = calib_row.get('Pearson')
                n_overlap = calib_row.get('N.Overlap')
                corr_subtitle = (f"Model ({status}): R²={r2:.3f}, MAE={mae:.3f}, "
                                 f"Pearson={pearson:.3f}, N={int(n_overlap)}")
            else:
                # recovered values are predicted from JUMPlion, so only the
                # observed (real DIA-NN) points are used for the correlation
                observed = merged[merged['Status'] == 'observed']
                pearson_r = observed['jumplion_log2'].corr(observed['diann_log2'])
                corr_subtitle = (f"No calibration model for this precursor; "
                                 f"Pearson r = {pearson_r:.3f} (n={len(observed)}, observed only)")
        else:
            corr_subtitle = "No overlapping runs with valid quantities"

        corr_fig.update_layout(
            title=dict(text=f"JUMPlion (MS1_MS2) vs. Final Quantification<br><sup>{corr_subtitle}</sup>"),
            xaxis_title="Log2(JUMPlion MS1_MS2 Intensity)",
            yaxis_title="Log2(Final Intensity)",
            template=template_name,
            height=500,
            xaxis=dict(domain=[0, 0.8], range=xaxis_range, constrain='domain', dtick=common_dtick),
            yaxis=dict(range=yaxis_range, scaleanchor='x', scaleratio=1, constrain='domain', dtick=common_dtick),
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

        return [
            dcc.Graph(figure=jumplion_quant_fig, config=to_image_config),
            dcc.Graph(figure=diann_quant_fig, config=to_image_config),
            dcc.Graph(figure=corr_fig, config=to_image_config),
        ]
    except Exception as e:
        logger.error(f"Error updating plot: {str(e)}")
        return [html.Div(f"Error updating plot: {str(e)}")]


@app.callback(
    [Output('run-dropdown', 'options'),
     Output('stored-prec-report', 'data')],     
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
                         'RT.Apex', 'RT.Left', 'RT.Right', 'Reference.XIC', 'Reference.XIC.Type',
                         'Reference.XIC.Score', #'Strategy',
                         'Evidence.Tier', 'Final.Status',
                         'MS1.XIC.Info', 'MS2.XIC.Info',
                         'MS2.XIC.Corr', 'MS2.XIC.Quant', 'MS2.XIC.Quant.Adjusted',
                         'MS1.XIC.Corr', 'MS1.XIC.Quant', 'MS1.XIC.Quant.Adjusted',
                         'MS1.Rel.Intensity', 'MS2.Rel.Intensity']
        df = pd.read_parquet(data_path, 
                             columns=required_cols,
                             filters=[('Precursor.Id', '==', selected_precursor)])
        if df.empty:
            return [], None
        else:
            run_names = df['Run'].unique().tolist()
            run_names_simple = extract_unique_parts(run_names)
            run_options = [{'label': run_names_simple[i], 'value': run} for i, run in enumerate(run_names)]
            return run_options, df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error updating run options: {e}")
        return [], None

def _resolve_reference_xic(run_report_row: pd.Series):
    """Resolve the reference XIC's level, channel name/index, and score from
    the report row. Reference.XIC is a 0-based index into whichever level's
    XIC.Info list (MS1 or MS2) Reference.XIC.Type points to (mirrors the logic
    used in the Individual Run tab)."""
    ref_type = run_report_row.get('Reference.XIC.Type')
    ref_idx = run_report_row.get('Reference.XIC')
    score_val = run_report_row.get('Reference.XIC.Score')
    ref_score = float(score_val) if pd.notna(score_val) else None
    if ref_type not in ('ms1', 'ms2') or pd.isna(ref_idx):
        return None, None, None, ref_score
    ref_idx = int(ref_idx)
    info_col = 'MS1.XIC.Info' if ref_type == 'ms1' else 'MS2.XIC.Info'
    xic_names = run_report_row[info_col].split(';')
    if ref_idx < 0 or ref_idx >= len(xic_names):
        return ref_type, None, None, ref_score
    return ref_type, ref_idx, xic_names[ref_idx].split('/')[0], ref_score


def _load_one_run_xic(run: str, out_int_dir: Path, selected_precursor: str,
                       level: str, run_report_row: pd.Series):
    """Read one run's raw XIC (I/O-bound) plus its report-derived markers.
    Runs inside a thread pool, so it must not touch shared mutable state."""
    xic_file_path = get_xic_path(out_int_dir, run)
    if not xic_file_path.exists():
        return None
    df = pl.scan_parquet(xic_file_path).filter(
        pl.col("Precursor.Id") == selected_precursor
    ).collect()
    if df.height == 0:
        return None
    row = df.row(0, named=True)
    #print(f"Precursor Index in library: {row['prec_index']}")

    if level == 'ms2':
        rt = np.array(row["ms2_rt"])
        values = np.array(row["ms2"])
        n_channels = row["n_prods"]
        xic_names = run_report_row['MS2.XIC.Info'].split(';')
        colors = [ms2_colors[i] for i in range(n_channels)]
    else:
        rt = np.array(row["ms1_rt"])
        values = np.array(row["ms1"])
        n_channels = row["n_isotopes"]
        xic_names = run_report_row['MS1.XIC.Info'].split(';')
        colors = [ms1_colors[name.split('/')[0]] for name in xic_names]

    ch_names = [xic_names[ch].split('/')[0] if ch < len(xic_names) else f"ch{ch}"
                for ch in range(n_channels)]

    rt_apex_val = run_report_row.get('RT.Apex')
    rt_left_val = run_report_row.get('RT.Left')
    rt_right_val = run_report_row.get('RT.Right')

    evidence_tier_val = run_report_row.get('Evidence.Tier')
    evidence_tier = str(evidence_tier_val) if pd.notna(evidence_tier_val) else None
    final_status_val = run_report_row.get('Final.Status')
    final_status = final_status_val if pd.notna(final_status_val) else None

    ref_type, ref_idx, ref_name, ref_score = _resolve_reference_xic(run_report_row)

    #strategy_state_val = run_report_row.get('Strategy')
    #strategy_state = str(strategy_state_val) if pd.notna(strategy_state_val) else None

    def _clean_num(v):
        return float(v) if v is not None and pd.notna(v) else None

    def _clean_idx(v):
        return int(v) if v is not None and pd.notna(v) and int(v) >= 0 else None

    all_ms1_names = run_report_row['MS1.XIC.Info'].split(';')
    all_ms2_names = run_report_row['MS2.XIC.Info'].split(';')

    def _resolve_combined_name(idx, names):
        if idx is None:
            return None
        return names[idx].split('/')[0]

    best_ms1_xic_index = _clean_idx(row.get('best_ms1_xic_index'))
    best_ms2_xic_index = _clean_idx(row.get('best_ms2_xic_index'))

    return {
        'rt': rt.tolist(),
        'values': values.tolist(),
        'ch_names': ch_names,
        'colors': colors,
        'rt_markers': {
            'apex': float(rt_apex_val) if pd.notna(rt_apex_val) else None,
            'left': float(rt_left_val) if pd.notna(rt_left_val) else None,
            'right': float(rt_right_val) if pd.notna(rt_right_val) else None,
        },
        'ref_xic_type': ref_type,
        'ref_xic_idx': ref_idx,
        'ref_xic_name': ref_name,
        'ref_xic_score': ref_score,
        'best_ms1_xic_score': _clean_num(row.get('best_ms1_xic_score')),
        'best_ms2_xic_score': _clean_num(row.get('best_ms2_xic_score')),
        'best_ms1_xic_name': _resolve_combined_name(best_ms1_xic_index, all_ms1_names),
        'best_ms2_xic_name': _resolve_combined_name(best_ms2_xic_index, all_ms2_names),
        'evidence_tier': evidence_tier,
        'final_status': final_status,
    }


@app.callback(
    Output('xic-all-raw-store', 'data'),
    Input('stored-prec-report', 'data'),
    Input('xic-all-level', 'value'),
    Input('main-tabs', 'active_tab'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
    State('xic-all-raw-store', 'data'),
    prevent_initial_call=True
)
def load_all_runs_raw_data(stored_prec_report: List[Dict[str, Any]],
                            xic_level: str,
                            active_tab: str,
                            outdir_path: str,
                            stored_meta: List[Dict[str, Any]],
                            current_store: Dict[str, Any]):
    # Expensive stage: reads every run's raw XIC parquet file (in parallel,
    # since it's I/O-bound). Selecting a precursor alone must not trigger
    # this — it only runs once the All Runs tab is actually open (either
    # because the user just switched to it, or because the precursor/level
    # changed while already on it), so the dcc.Loading wrapper around that
    # tab's content is the only loading feedback the user sees.
    if active_tab != 'tab-all-runs':
        return no_update

    if not stored_prec_report or not outdir_path:
        return None

    try:
        prec_report_df = pd.DataFrame(stored_prec_report)
        selected_precursor = prec_report_df['Precursor.Id'].iloc[0]
        level = xic_level if xic_level in ('ms1', 'ms2') else 'ms1'

        # Re-entering the tab (without actually changing precursor or MS
        # level) shouldn't re-touch disk — only a real selection change
        # should trigger a reload.
        cached = current_store or {}
        if cached.get('precursor') == selected_precursor and cached.get('level') == level:
            return no_update

        out_int_dir = Path(outdir_path) / 'intermediate'
        meta_df = pd.DataFrame(stored_meta) if stored_meta else pd.DataFrame()
        has_group = not meta_df.empty and 'Group' in meta_df.columns
        available_runs = set(prec_report_df['Run'].unique().tolist())

        # Build (group, run, run_label) list, ordered by the meta file when available
        run_entries = []
        if not meta_df.empty and 'Run' in meta_df.columns:
            for _, meta_row in meta_df.iterrows():
                run = meta_row['Run']
                if run not in available_runs:
                    continue
                group = meta_row['Group'] if has_group and pd.notna(meta_row.get('Group')) else 'All Runs'
                run_label = meta_row.get('Run_simp', run)
                run_entries.append((group, run, run_label))
        else:
            run_names = prec_report_df['Run'].unique().tolist()
            run_names_simple = extract_unique_parts(run_names)
            run_entries = [('All Runs', run, label) for run, label in zip(run_names, run_names_simple)]

        if not run_entries:
            return None

        # Initial (pre-refinement) predicted RT per run, used to mark the
        # center of the extraction window on each XIC plot.
        rt_pred_row = None
        rt_pred_path = out_int_dir / 'rt_prediction' / 'rt_mat_after_prediction.tsv'
        if rt_pred_path.exists():
            try:
                rt_pred_df = pd.read_csv(rt_pred_path, sep='\t', index_col=0)
                if selected_precursor in rt_pred_df.index:
                    rt_pred_row = rt_pred_df.loc[selected_precursor]
            except Exception as rt_err:
                logger.error(f"Error loading RT prediction file: {rt_err}")

        # Group runs, preserving order of first appearance
        groups_order = []
        grouped_runs = {}
        for group, run, run_label in run_entries:
            if group not in grouped_runs:
                grouped_runs[group] = []
                groups_order.append(group)
            grouped_runs[group].append([run, run_label])

        unique_runs = list(dict.fromkeys(run for _, run, _ in run_entries))
        run_report_rows = {
            run: prec_report_df.loc[prec_report_df['Run'] == run].iloc[0]
            for run in unique_runs if (prec_report_df['Run'] == run).any()
        }

        def load_one(run):
            try:
                run_report_row = run_report_rows.get(run)
                if run_report_row is None:
                    return run, None
                return run, _load_one_run_xic(run, out_int_dir, selected_precursor, level, run_report_row)
            except Exception as run_err:
                logger.error(f"Error loading XIC for run {run}: {run_err}")
                return run, None

        per_run = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(unique_runs)))) as executor:
            for run, data in executor.map(load_one, unique_runs):
                if data is None:
                    continue
                initial_rt = None
                if rt_pred_row is not None and run in rt_pred_row.index:
                    val = rt_pred_row[run]
                    if pd.notna(val):
                        initial_rt = float(val)
                data['initial_rt'] = initial_rt
                per_run[run] = data

        store_payload = {
            'precursor': selected_precursor,
            'groups_order': groups_order,
            'grouped_runs': grouped_runs,
            'per_run': per_run,
            'level': level,
        }
        return store_payload
    except Exception as e:
        logger.error(f"Error loading all-runs XIC data: {e}")
        return None


@app.callback(
    Output('xic-all-runs-container', 'children'),
    Input('xic-all-raw-store', 'data'),
    Input('xic-all-radius', 'value'),
    Input('prec-diann-quant', 'data'),
    Input(ThemeSwitchAIO.ids.switch(THEME_SWITCH_AIO_ID), 'value'),
    prevent_initial_call=True
)
def render_all_runs_xic_plot(raw_store: Dict[str, Any],
                              xic_radius,
                              prec_diann_quant: List[Dict[str, Any]],
                              light_theme):
    template_name = PLOTLY_TEMPLATE_LIGHT if light_theme else PLOTLY_TEMPLATE_DARK
    # Cheap stage: only slices/redraws already-loaded arrays, so radius and
    # DIA-NN-status changes never re-touch disk.
    if not raw_store or not raw_store.get('per_run'):
        return [html.Div("Select a precursor to visualize its XICs across all runs.")]

    try:
        groups_order = raw_store['groups_order']
        grouped_runs = raw_store['grouped_runs']
        per_run = raw_store['per_run']
        level = raw_store.get('level', 'ms1')
        precursor = raw_store.get('precursor', '')

        # Legend badges are identical for every run/group at a given
        # precursor+level, so build them once from any available run's
        # channel names/colors rather than lazily inside the per-tile loop.
        legend_badges = []
        for _data in per_run.values():
            legend_badges = [html.Span(name, style={
                'backgroundColor': _data['colors'][ch], 'color': 'white',
                'padding': '2px 8px', 'borderRadius': '4px', 'marginRight': '6px',
                'fontSize': '12px', 'display': 'inline-block', 'flexShrink': 0,
            }) for ch, name in enumerate(_data['ch_names'])]
            break

        def build_combined_header(group_name):
            # Precursor id, legend, and group name all share ONE sticky row
            # (rebuilt per group so the group name updates as you scroll
            # between groups) instead of three stacked rows, so together
            # they only ever occupy a single row's worth of vertical space.
            return html.Div([
                html.Div(f"Precursor: {precursor}", style={
                    'fontSize': '16px', 'fontWeight': 'bold', 'whiteSpace': 'nowrap',
                    'marginRight': '15px', 'flexShrink': 0,
                }),
                html.Div(legend_badges, style={
                    'display': 'flex', 'flexWrap': 'nowrap', 'overflowX': 'auto',
                    'flex': '1 1 auto', 'minWidth': 0,
                }),
                html.Div(f"Group: {group_name}", style={
                    'fontSize': '14px', 'fontWeight': 'bold', 'whiteSpace': 'nowrap',
                    'marginLeft': '15px', 'flexShrink': 0,
                }),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'position': 'sticky', 'top': ALL_RUNS_HEADER_STICKY_TOP,
                'zIndex': 1005, 'backgroundColor': 'var(--bs-body-bg, #fff)',
                'paddingTop': '10px', 'paddingBottom': '10px',
            })

        diann_row = None
        if prec_diann_quant:
            diann_df = pd.DataFrame(prec_diann_quant)
            if not diann_df.empty:
                diann_row = diann_df.iloc[0]

        def diann_status(run_label):
            return diann_reported_status(diann_row, run_label)

        radius = None
        if xic_radius not in (None, ''):
            try:
                radius = max(int(xic_radius), 0)
            except (ValueError, TypeError):
                radius = None

        # CSS grid wraps automatically to the container width, so the number of
        # tiles per row adjusts dynamically as the browser window is resized.
        grid_style = {
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fill, minmax(260px, 1fr))',
            'gap': '10px',
        }
        graph_config = {**to_image_config, 'responsive': True}

        group_blocks = []
        for group in groups_order:
            runs_in_group = grouped_runs.get(group, [])

            # First pass: slice the cached arrays and track the group-wide
            # y-range so all tiles in the group share one scale.
            run_data_list = []
            y_min, y_max = None, None
            for run, run_label in runs_in_group:
                data = per_run.get(run)
                if data is None:
                    continue

                rt = np.array(data['rt'])
                values = np.array(data['values'])
                ch_names = data['ch_names']
                colors = data['colors']
                rt_markers = data['rt_markers']
                initial_rt = data.get('initial_rt')
                ref_xic_type = data.get('ref_xic_type')
                ref_xic_idx = data.get('ref_xic_idx')
                ref_xic_name = data.get('ref_xic_name')
                ref_xic_score = data.get('ref_xic_score')
                best_ms1_xic_score = data.get('best_ms1_xic_score')
                best_ms2_xic_score = data.get('best_ms2_xic_score')
                best_ms1_xic_name = data.get('best_ms1_xic_name')
                best_ms2_xic_name = data.get('best_ms2_xic_name')
                evidence_tier = data.get('evidence_tier')
                final_status = data.get('final_status')
                #strategy_state = data.get('strategy_state')

                if radius is not None and rt_markers['apex'] is not None and len(rt) > 0:
                    apex_idx = int(np.argmin(np.abs(rt - rt_markers['apex'])))
                    lo = max(0, apex_idx - radius)
                    hi = min(len(rt), apex_idx + radius + 1)
                    rt = rt[lo:hi]
                    values = values[lo:hi, :]

                finite_vals = values[np.isfinite(values)]
                if finite_vals.size:
                    cur_min, cur_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                    y_min = cur_min if y_min is None else min(y_min, cur_min)
                    y_max = cur_max if y_max is None else max(y_max, cur_max)

                run_data_list.append((run_label, rt, values, ch_names, colors, initial_rt,
                                      diann_status(run_label), rt_markers,
                                      ref_xic_type, ref_xic_idx, ref_xic_name,
                                      ref_xic_score, best_ms1_xic_score, best_ms2_xic_score,
                                      best_ms1_xic_name, best_ms2_xic_name,
                                      evidence_tier, final_status))

            if not run_data_list:
                continue

            if y_min is not None and y_max is not None:
                pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_max), 1) * 0.05
                y_range = [y_min - pad, y_max + pad]
            else:
                y_range = None

            # Second pass: build one tile per run, sharing the group's y-range.
            run_tiles = []
            for (run_label, rt, values, ch_names, colors, initial_rt, diann_stat, rt_markers,
                 ref_xic_type, ref_xic_idx, ref_xic_name, ref_xic_score,
                 best_ms1_xic_score, best_ms2_xic_score,
                 best_ms1_xic_name, best_ms2_xic_name,
                 evidence_tier, final_status) in run_data_list:
                # Highlight whichever channel is the best-scoring XIC for the
                # currently displayed level (matched by ion name, since the
                # best index is reported against the combined MS1+MS2 array).
                best_name_for_level = best_ms1_xic_name if level == 'ms1' else best_ms2_xic_name
                highlight_ch = (ch_names.index(best_name_for_level)
                                if best_name_for_level is not None and best_name_for_level in ch_names
                                else None)

                fig = go.Figure()
                for ch in range(values.shape[1]):
                    is_optimal = ch == highlight_ch
                    fig.add_trace(go.Scatter(
                        x=rt,
                        y=values[:, ch],
                        mode='lines+markers',
                        name=ch_names[ch] + (" (best)" if is_optimal else ""),
                        line=dict(color=colors[ch], width=3 if is_optimal else 2),
                        marker=dict(color=colors[ch], size=6 if is_optimal else 4),
                        showlegend=False,
                    ))
                if rt_markers['left'] is not None:
                    fig.add_vline(x=rt_markers['left'], line=dict(color="gray", dash="dash"))
                if rt_markers['right'] is not None:
                    fig.add_vline(x=rt_markers['right'], line=dict(color="gray", dash="dash"))
                if rt_markers['apex'] is not None:
                    fig.add_vline(x=rt_markers['apex'], line=dict(color="red", dash="dash"))
                if initial_rt is not None:
                    fig.add_vline(x=initial_rt, line=dict(color="green", dash="dash"))
                fig.update_layout(
                    title=dict(text=run_label, font=dict(size=12)),
                    template=template_name,
                    height=240,
                    margin=dict(l=40, r=10, t=35, b=30),
                )
                fig.update_yaxes(tickformat=".1e", range=y_range)

                tile_children = [dcc.Graph(figure=fig, config=graph_config, style={'height': '240px'})]
                # Tier/Status are only useful for runs DIA-NN didn't already
                # report — for "reported" runs the badges are hidden (not
                # removed) so every tile in the grid keeps the same height.
                tier_text = f"Tier: {evidence_tier}" if evidence_tier is not None else "Tier: NA"
                status_text = f"Status: {final_status}" if final_status is not None else "Status: NA"
                tile_children.insert(0, html.Div([
                    html.Div(tier_text, style={
                        'backgroundColor': '#6f42c1', 'color': 'white',
                        'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                        'padding': '1px 0', 'borderRadius': '3px', 'flex': '1',
                    }),
                    html.Div(status_text, style={
                        'backgroundColor': '#17a2b8', 'color': 'white',
                        'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                        'padding': '1px 0', 'borderRadius': '3px', 'flex': '1',
                    }),
                ], style={'display': 'flex', 'gap': '2px', 'marginBottom': '2px',
                          'visibility': 'hidden' if diann_stat == 'reported' else 'inherit'}))
                # if best_ms1_xic_score is not None or best_ms2_xic_score is not None:
                #     ms1_best_text = (f"MS1 best: {best_ms1_xic_name}, {best_ms1_xic_score:.2f}"
                #                       if best_ms1_xic_score is not None else "MS1 best: NA")
                #     ms2_best_text = (f"MS2 best: {best_ms2_xic_name}, {best_ms2_xic_score:.2f}"
                #                       if best_ms2_xic_score is not None else "MS2 best: NA")
                #     badge_style = {
                #         'backgroundColor': '#17a2b8', 'color': 'white',
                #         'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                #         'padding': '1px 0', 'borderRadius': '3px', 'flex': '1',
                #     }
                #     tile_children.insert(0, html.Div([
                #         html.Div(ms1_best_text, style=badge_style),
                #         html.Div(ms2_best_text, style=badge_style),
                #     ], style={'display': 'flex', 'gap': '2px', 'marginBottom': '2px'}))
                # if ref_xic_type is not None:
                #     ref_text = f"Reference XIC: {ref_xic_type.upper()}"
                #     if ref_xic_name is not None:
                #         ref_text += f" · {ref_xic_name}"
                #     if ref_xic_score is not None:
                #         ref_text += f" · {ref_xic_score:.2f}"
                #     tile_children.insert(0, html.Div(ref_text, style={
                #         'backgroundColor': '#6f42c1', 'color': 'white',
                #         'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                #         'padding': '1px 0', 'borderRadius': '3px', 'marginBottom': '2px',
                #     }))
                # if strategy_state is not None:
                #     tile_children.insert(0, html.Div(f"Strategy: {strategy_state}", style={
                #         'backgroundColor': '#fd7e14', 'color': 'white',
                #         'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                #         'padding': '1px 0', 'borderRadius': '3px', 'marginBottom': '2px',
                #     }))
                if diann_stat is not None:
                    badge_color = '#2ca02c' if diann_stat == 'reported' else '#d62728'
                    tile_children.insert(0, html.Div(diann_stat, style={
                        'backgroundColor': badge_color, 'color': 'white',
                        'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center',
                        'padding': '1px 0', 'borderRadius': '3px', 'marginBottom': '2px',
                    }))

                run_tiles.append(html.Div(tile_children))

            group_blocks.append(
                # position:relative makes this div the sticky header's
                # "containing block" for push-off purposes, so the header is
                # only ever stuck within its OWN group's vertical span and
                # gets cleanly carried off-screen once that span scrolls
                # past — instead of remaining visible/overlapping once the
                # next group's header has already reached the same spot.
                html.Div([
                    build_combined_header(group),
                    html.Div(run_tiles, style=grid_style),
                ], style={'position': 'relative'})
            )

        if not group_blocks:
            return [html.Div("No XIC data found for the selected precursor.")]

        children = group_blocks
        return children
    except Exception as e:
        logger.error(f"Error updating all-runs XIC plot: {e}")
        return [html.Div(f"Error updating plot: {str(e)}")]


@app.callback(
    [Output('show-precid-i', 'children'),
     Output('show-run-info', 'children'),
     Output('xic-individual-container', 'children')],
    Input('precursor-dropdown', 'value'),
    Input('run-dropdown', 'value'),
    Input('show-ms1-inter', 'value'),
    Input('show-reference', 'value'),
    Input('show-cosine', 'value'),
    Input(ThemeSwitchAIO.ids.switch(THEME_SWITCH_AIO_ID), 'value'),
    Input('stored-prec-report', 'data'),
    State('stored-outdir', 'data'),
    State('stored-meta', 'data'),
    State('prec-diann-quant', 'data')
)
def update_XIC_plots_table(selected_precursor: str,
                           selected_run: str,
                           show_interpolated_ms1: bool,
                           show_optimal: bool,
                           show_cosine: bool,
                           light_theme,
                           stored_prec_report: List[Dict[str, Any]],
                           outdir_path: str,
                           stored_meta: List[Dict[str, Any]],
                           prec_diann_quant: List[Dict[str, Any]]) -> tuple:
    template = PLOTLY_TEMPLATE_LIGHT if light_theme else PLOTLY_TEMPLATE_DARK
    if selected_run is None or selected_precursor is None or not stored_prec_report:
        return '', '', [html.Div("Select a precursor and run to view XICs.")]

    try:
        # Extract relevant data
        prec_title = f"     {selected_precursor}"

        # DIA-NN reported/missing status for this run, based on whether
        # Precursor.Quantity is present (keyed by simplified run name).
        diann_stat = None
        if prec_diann_quant and stored_meta:
            diann_df = pd.DataFrame(prec_diann_quant)
            meta_df = pd.DataFrame(stored_meta)
            if not diann_df.empty and not meta_df.empty and 'Run' in meta_df.columns:
                match = meta_df.loc[meta_df['Run'] == selected_run, 'Run_simp']
                if not match.empty:
                    diann_stat = diann_reported_status(diann_df.iloc[0], match.iloc[0])

        run_title_children = [f"     Run: {selected_run}"]
        if diann_stat is not None:
            badge_color = '#2ca02c' if diann_stat == 'reported' else '#d62728'
            run_title_children.append(html.Span(diann_stat, style={
                'backgroundColor': badge_color, 'color': 'white', 'fontSize': '12px',
                'fontWeight': 'bold', 'padding': '2px 8px', 'borderRadius': '4px',
                'marginLeft': '10px', 'verticalAlign': 'middle',
            }))
        run_title = html.Div(run_title_children)

        # Get XIC file path using absolute paths
        out_int_dir = Path(outdir_path) / 'intermediate'

        # Initial (pre-refinement) predicted RT for this run, used to mark the
        # center of the extraction window on the raw XIC plots.
        initial_rt = None
        rt_pred_path = out_int_dir / 'rt_prediction' / 'rt_mat_after_prediction.tsv'
        if rt_pred_path.exists():
            try:
                rt_pred_df = pd.read_csv(rt_pred_path, sep='\t', index_col=0)
                if selected_precursor in rt_pred_df.index and selected_run in rt_pred_df.columns:
                    val = rt_pred_df.loc[selected_precursor, selected_run]
                    if pd.notna(val):
                        initial_rt = float(val)
            except Exception as rt_err:
                logger.error(f"Error loading RT prediction file: {rt_err}")

        xic_file_path = get_xic_path(out_int_dir, selected_run)
        if not xic_file_path.exists():
            raise FileNotFoundError(f"XIC file not found: {xic_file_path}")
        df = pl.scan_parquet(xic_file_path).filter(pl.col("Precursor.Id") == selected_precursor).collect()
        row = df.filter(pl.col("Precursor.Id") == selected_precursor)
        if row.height == 0:
            raise ValueError(f"No XIC data found for precursor {selected_precursor}")
        row = row.row(0, named=True)
        
        # stored-prec-report is refreshed by a separate callback (update_run_options)
        # triggered by the same precursor-dropdown change, so it can briefly still
        # hold the PREVIOUS precursor's rows here. Filtering on Precursor.Id too
        # (not just Run) avoids mixing a stale precursor's report row with this
        # (already-correct) precursor's raw XIC arrays. Once stored-prec-report
        # catches up, this callback re-fires (it's an Input) and renders correctly.
        prec_report_df = pd.DataFrame(stored_prec_report)
        if 'Precursor.Id' in prec_report_df.columns:
            prec_report_df = prec_report_df[prec_report_df['Precursor.Id'] == selected_precursor]
        prec_data = prec_report_df[prec_report_df['Run'] == selected_run].reset_index(drop=True)
        if prec_data.empty:
            raise ValueError("No data found for selected run")
        
        # Precursor info table
        prec_info = prec_data[['MS1.XIC.Info',
                               'MS1.XIC.Corr',
                               'MS1.XIC.Quant',
                               'MS1.XIC.Quant.Adjusted',
                               'MS1.Rel.Intensity']].iloc[0]
        ms1_xics = prec_info['MS1.XIC.Info'].split(';')
        ms1_xic_names = []
        ms1_xic_mzs = []
        for ms1_xic in ms1_xics:
            parts = ms1_xic.split('/')
            ms1_xic_names.append(parts[0])
            ms1_xic_mzs.append(parts[1])

        prec_info_df = pd.DataFrame({
            "Type": ["MS1"] * len(ms1_xics),
            "Name": ms1_xic_names,
            "m/z": ms1_xic_mzs,
            "Reference Corr": string2float(prec_info['MS1.XIC.Corr'].split(';'), 2),
            "XIC Quant": string2float(prec_info['MS1.XIC.Quant'].split(';'), 2),
            "XIC Quant (Adj)": string2float(prec_info['MS1.XIC.Quant.Adjusted'].split(';'), 2)
        })

        # Fragment info table
        frag_info = prec_data[['MS2.XIC.Info',
                               'MS2.XIC.Corr',
                               'MS2.XIC.Quant',
                               'MS2.XIC.Quant.Adjusted',
                               'MS2.Rel.Intensity']].iloc[0]
        ms2_xics = frag_info['MS2.XIC.Info'].split(';')
        ms2_xic_names = []
        ms2_xic_mzs = []
        for ms2_xic in ms2_xics:
            parts = ms2_xic.split('/')
            ms2_xic_names.append(parts[0])
            ms2_xic_mzs.append(parts[1])
        
        frag_info_df = pd.DataFrame({
            "Type": ["MS2"] * len(ms2_xics),
            "Name": ms2_xic_names,
            "m/z": ms2_xic_mzs,
            "Reference Corr": string2float(frag_info['MS2.XIC.Corr'].split(';'), 2),
            "XIC Quant": string2float(frag_info['MS2.XIC.Quant'].split(';'), 2),
            "XIC Quant (Adj)": string2float(frag_info['MS2.XIC.Quant.Adjusted'].split(';'), 2)
        })

        def _fmt_cosine(v):
            return f"{v:.3f}" if v is not None and pd.notna(v) else "NA"

        # Cosine similarities are computed on the fly (only when requested)
        # from the observed quantities and the theoretical relative intensities.
        if show_cosine:
            ms1_cos_text = (f"<br><sup>Isotope Cosine: "
                            f"{_fmt_cosine(cosine_similarity_vectors(prec_info['MS1.XIC.Quant'], prec_info['MS1.Rel.Intensity']))}"
                            f"</sup>")
            ms2_cos_text = (f"<br><sup>Fragment Cosine: "
                            f"{_fmt_cosine(cosine_similarity_vectors(frag_info['MS2.XIC.Quant'], frag_info['MS2.Rel.Intensity']))} "
                            f"&nbsp;|&nbsp; Top5: "
                            f"{_fmt_cosine(topk_cosine_similarity(frag_info['MS2.XIC.Quant'], frag_info['MS2.Rel.Intensity'], k=5))}"
                            f"</sup>")
        else:
            ms1_cos_text = ms2_cos_text = ""

        def make_stems(x_values, values):
            # Interleave (x, 0), (x, value), (x, None) per ion so Plotly
            # draws a disconnected vertical stick at each x position instead
            # of connecting neighboring ions with a diagonal line.
            x_stems, y_stems = [], []
            for x, val in zip(x_values, values):
                x_stems.extend([x, x, x])
                y_stems.extend([0, val, None])
            return x_stems, y_stems

        def build_mirror_fig(names, mz_strings, real_raw_str, theo_rel_str, title, legend_visible):
            mzs = parse_floats(mz_strings)
            real_pct = normalize_to_pct(parse_floats(real_raw_str.split(';')))
            theo_pct = normalize_to_pct(parse_floats(theo_rel_str.split(';')))
            theo_neg = [-v for v in theo_pct]
            real_x, real_y = make_stems(mzs, real_pct)
            theo_x, theo_y = make_stems(mzs, theo_neg)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=real_x, y=real_y, mode='lines', name='Observed',
                line=dict(color='steelblue', width=2),
            ))
            fig.add_trace(go.Scatter(
                x=theo_x, y=theo_y, mode='lines', name='Theoretical',
                line=dict(color='darkorange', width=2),
            ))
            fig.add_trace(go.Scatter(
                x=mzs, y=theo_neg, mode='text', text=names, textposition='bottom center',
                textfont=dict(size=9, color='darkorange'), showlegend=False, hoverinfo='skip',
            ))
            fig.add_hline(y=0, line=dict(color='black', width=1))

            if mzs:
                mz_min, mz_max = min(mzs), max(mzs)
                pad = (mz_max - mz_min) * 0.05 if mz_max > mz_min else max(abs(mz_max), 1) * 0.05
                x_range = [mz_min - pad, mz_max + pad]
            else:
                x_range = None

            # Always reserve legend space (so MS1/MS2 plot areas line up), but
            # make the legend itself invisible when legend_visible is False.
            legend = dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
                           font=dict(size=9))
            if not legend_visible:
                legend.update(
                    font=dict(size=9, color='rgba(0,0,0,0)'),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    itemclick=False,
                    itemdoubleclick=False,
                )

            fig.update_layout(
                title=dict(text=title, font=dict(size=13)),
                height=400,
                autosize=True,
                xaxis_title="m/z",
                xaxis=dict(range=x_range),
                yaxis_title="Relative Intensity (%)",
                yaxis=dict(range=[-110, 110]),
                showlegend=True,
                legend=legend,
                template=template,
                margin=dict(l=0, r=0, t=70, b=70),
            )
            return fig

        fig_ms1_mirror = build_mirror_fig(
            ms1_xic_names, ms1_xic_mzs, prec_info['MS1.XIC.Quant'], prec_info['MS1.Rel.Intensity'],
            f"MS1 Real vs. Theoretical Intensity{ms1_cos_text}",
            legend_visible=False
        )
        fig_ms2_mirror = build_mirror_fig(
            ms2_xic_names, ms2_xic_mzs, frag_info['MS2.XIC.Quant'], frag_info['MS2.Rel.Intensity'],
            f"MS2 Real vs. Theoretical Intensity{ms2_cos_text}",
            legend_visible=True
        )

        ref_xic_type = prec_data['Reference.XIC.Type'].iloc[0]
        ref_xic = prec_data['Reference.XIC'].iloc[0]

        # best_ms1_xic_index is local to the MS1 isotope array (0..n_isotopes-1)
        # and best_ms2_xic_index is local to the MS2 fragment array (0..n_prods-1).
        def _clean_best_idx(v):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                return int(v)
            except (TypeError, ValueError):
                return None

        best_ms1_idx = _clean_best_idx(row.get('best_ms1_xic_index'))
        best_ms2_idx = _clean_best_idx(row.get('best_ms2_xic_index'))

        # annotate reference/best ions in the info tables (guarded: an index
        # can fall outside the reported ion table for runs where the stored
        # index/score arrays were not trimmed to match the final reported set)
        def _annotate_table(df, ref_idx, best_idx, level_label):
            for idx in {i for i in (ref_idx, best_idx) if i is not None}:
                if idx not in df.index:
                    logger.error(f"Index {idx} out of range for {selected_precursor}/{selected_run} "
                                 f"({level_label} table has {len(df)} rows)")
                    continue
                tags = []
                if idx == ref_idx:
                    tags.append("reference")
                if idx == best_idx:
                    tags.append("best")
                df.loc[idx, 'Name'] = df.loc[idx, 'Name'] + f" ({', '.join(tags)})"

        _annotate_table(prec_info_df, ref_xic if ref_xic_type == 'ms1' else None, best_ms1_idx, "MS1")
        _annotate_table(frag_info_df, ref_xic if ref_xic_type == 'ms2' else None, best_ms2_idx, "MS2")

        n_isotopes, n_prods = row["n_isotopes"], row["n_prods"]

        def _highlight_suffix(is_ref, is_best):
            tags = []
            if is_ref:
                tags.append("reference")
            if is_best:
                tags.append("best")
            return f" ({', '.join(tags)})" if tags else ""

        ms1_rt = np.array(row["ms1_rt"])
        ms2_rt = np.array(row["ms2_rt"])
        ms1 = np.array(row["ms1"])      # shape: (n_frames, n_isotopes)
        ms2 = np.array(row["ms2"])      # shape: (n_frames, n_prods)
        ms1_i = np.array(row["ms1_i"])  # shape: (n_frames, n_isotopes)

        # print(ms1_rt)
        # print(ms1)
        # print(ms2_rt)
        # print(ms2)
        # print(ms1_i)

        ms1_traces = []
        for i in range(n_isotopes):
            is_ref = ref_xic_type == 'ms1' and i == ref_xic
            is_best = best_ms1_idx is not None and i == best_ms1_idx
            highlight = is_ref or is_best
            ms1_traces.append(go.Scatter(
                x=ms1_rt,
                y=ms1[:, i],
                mode='lines+markers',
                name=ms1_xics[i] + _highlight_suffix(is_ref, is_best),
                line=dict(color=ms1_colors[ms1_xics[i].split("/")[0]], width=3 if highlight else 2),
                marker=dict(color=ms1_colors[ms1_xics[i].split("/")[0]], size=8 if highlight else 6)
            ))

        ms2_traces = []
        for i in range(n_prods):
            is_ref = ref_xic_type == 'ms2' and i == ref_xic
            is_best = best_ms2_idx is not None and i == best_ms2_idx
            highlight = is_ref or is_best
            ms2_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms2[:, i],
                mode='lines+markers',
                name=ms2_xics[i] + _highlight_suffix(is_ref, is_best),
                line=dict(color=ms2_colors[i], width=3 if highlight else 2),
                marker=dict(color=ms2_colors[i], size=8 if highlight else 6)
            ))

        ms1_i_traces = []
        for i in range(n_isotopes):
            is_ref = ref_xic_type == 'ms1' and i == ref_xic
            is_best = best_ms1_idx is not None and i == best_ms1_idx
            highlight = is_ref or is_best
            ms1_i_traces.append(go.Scatter(
                x=ms2_rt,
                y=ms1_i[:, i],
                mode='lines+markers',
                name=ms1_xics[i] + _highlight_suffix(is_ref, is_best),
                line=dict(color=ms1_colors[ms1_xics[i].split("/")[0]], width=3 if highlight else 2),
                marker=dict(color=ms1_colors[ms1_xics[i].split("/")[0]], size=8 if highlight else 6)
            ))
            
        # The adjusted (smoothed/filtered) XIC file is only used here to source
        # the RT boundary marks (Apex/Left/Right) and the "reference" elution
        # trace, both of which are now overlaid directly on the raw MS1/MS2
        # plots below.
        xic_c_path = out_int_dir / "xic_files" / "xics.quant.parquet"
        df_c = pl.scan_parquet(xic_c_path).filter(
            (pl.col("Run") == selected_run) & (pl.col("Precursor.Id") == selected_precursor)
        ).collect()

        if df_c.height == 0:
            raise ValueError(f"No adjusted XIC data found for run={selected_run}, precursor={selected_precursor}")
        row_c = df_c.row(0, named=True)

        ms1_rt_c = np.array(row_c["ms1_rt"])
        ms2_rt_c = np.array(row_c["ms2_rt"])
        ms1_elution_c = np.array(row_c["ms1_elution"])
        ms2_elution_c = np.array(row_c["ms2_elution"])

        if show_optimal:
            #if ref_xic_type == 'ms2':
            ms2_traces.append(go.Scatter(
                x=ms2_rt_c,
                y=ms2_elution_c,
                mode='lines+markers',
                name='reference',
                marker=dict(size=6, color='red'),
                line=dict(dash='dash', color='red'),
            ))
            #elif ref_xic_type == 'ms1':
            ms1_traces.append(go.Scatter(
                x=ms2_rt_c,
                y=ms1_elution_c,
                mode='lines+markers',
                name='reference',
                marker=dict(size=6, color='red'),
                line=dict(dash='dash', color='red'),
            ))
            ms1_i_traces.append(go.Scatter(
                x=ms2_rt_c,
                y=ms1_elution_c,
                mode='lines+markers',
                name='reference',
                marker=dict(size=6, color='red'),
                line=dict(dash='dash', color='red'),
            ))

        def _fmt_corr(v):
            return f"{v:.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "NA"

        cross_ms1_ms2_corr = row.get("cross_ms1_ms2_corr")
        ms1_internal_corr = row.get("ms1_internal_corr")
        ms2_internal_corr = row.get("ms2_internal_corr")

        ms1_corr_subtitle = (f"MS1 internal corr: {_fmt_corr(ms1_internal_corr)}"
                              f" &nbsp;|&nbsp; Cross MS1/MS2 corr: {_fmt_corr(cross_ms1_ms2_corr)}")
        ms2_corr_subtitle = (f"MS2 internal corr: {_fmt_corr(ms2_internal_corr)}"
                              f" &nbsp;|&nbsp; Cross MS1/MS2 corr: {_fmt_corr(cross_ms1_ms2_corr)}")

        if not show_interpolated_ms1:
            fig_ms1 = go.Figure(data=ms1_traces)
        else:
            fig_ms1 = go.Figure(data=ms1_i_traces)

        fig_ms1.update_layout(
            title=dict(text=f"MS1-level XICs<br><sup>{ms1_corr_subtitle}</sup>"),
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
            title=dict(text=f"MS2-level XICs<br><sup>{ms2_corr_subtitle}</sup>"),
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
        
        # Add vertical lines to MS2 plot
        rt_marks = ['Apex', 'Left', 'Right']
        rt_boundaries = ms2_rt_c[[len(ms1_rt_c)//2, 0, -1]]
        for i, rt in enumerate(rt_boundaries):
            color = "red" if rt_marks[i] == 'Apex' else "gray"
            fig_ms1.add_vline(x=rt, line=dict(color=color, dash="dash"), annotation_text=rt_marks[i], annotation_position="top right")
            fig_ms2.add_vline(x=rt, line=dict(color=color, dash="dash"), annotation_text=rt_marks[i], annotation_position="top right")

        if initial_rt is not None:
            fig_ms1.add_vline(x=initial_rt, line=dict(color="green", dash="dash"),
                               annotation_text="", annotation_position="top left")
            fig_ms2.add_vline(x=initial_rt, line=dict(color="green", dash="dash"),
                               annotation_text="", annotation_position="top left")
        
        #print(row)
        #print(row["combined_score"])
        corr_score = np.array(row["corr_score"])
        shape_score = np.array(row["shape_score"])
        intensity_score = np.array(row["intensity_score"])
        xic_score = np.array(row["combined_score"])  # expected length n_isotopes + n_prods
        #print(xic_score)
        xic_info_df = pd.concat([prec_info_df, frag_info_df], axis=0)
        if len(xic_score) == len(xic_info_df):
            xic_info_df.insert(3, 'Correlation Score', np.round(corr_score, 2))
            xic_info_df.insert(4, 'Shape Score', np.round(shape_score, 2))
            xic_info_df.insert(5, 'Intensity Score', np.round(intensity_score, 2))
            xic_info_df.insert(6, 'Combined Score', np.round(xic_score, 2))
            #print(np.round(xic_score, 2))
        else:
            # Stored combined_score wasn't trimmed to match the final reported
            # ion set for this run/precursor; skip it rather than crash the tab.
            logger.error(f"combined_score length ({len(xic_score)}) does not match reported ion "
                         f"count ({len(xic_info_df)}) for {selected_precursor}/{selected_run}; "
                         f"omitting Combined Score column")
            xic_info_df.insert(3, 'Combined Score', np.nan)
    except Exception as e:
        logger.error(f"Error updating plot: {e}")
        return '', '', [html.Div(f"Error updating plot: {e}")]

    container_children = [
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ms1, config=to_image_config), width=11)
        ], style={'margin-top': '5px', 'margin-right': '10px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ms2, config=to_image_config), width=11)
        ], style={'margin-top': '5px', 'margin-right': '10px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ms1_mirror, config=to_image_config,
                              style={'height': '400px'}), width=4),
            dbc.Col(dcc.Graph(figure=fig_ms2_mirror, config=to_image_config,
                              style={'height': '400px'}), width=8)
        ], style={'margin-top': '5px', 'margin-right': '10px'}),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='xic-info-table',
                columns=XIC_INFO_TABLE_COLUMNS,
                data=xic_info_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'fontFamily': 'Arial, sans-serif',
                            'fontSize': '14px',
                            'textAlign': 'left',
                            'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                page_size=15,
            ), width=11, className="mx-auto"),
        ], style={'margin-top': '15px', 'margin-bottom': '20px'}),
    ]
    return prec_title, run_title, container_children

