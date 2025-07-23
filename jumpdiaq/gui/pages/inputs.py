# builtin
import os
import time
import platform
import subprocess
import queue
import threading
from pathlib import Path
# external
import dash
from dash import html, dcc, ctx, Input, Output, State
import dash_bootstrap_components as dbc
# local
from jumpdiaq.gui.app import app, page_sidebar_style, logger


sidebar = html.Div(
    [
        html.H3("Analyze"),
        html.Hr(),
        dbc.Nav(
            [   
                html.Label("Input MS files", style={"font-weight": "bold"}),
                dbc.Input(id='msfile-path', type='text', 
                          placeholder='Path of MS files (.raw, .mzML, .d)', 
                          style={'width': '100%'}, persistence=True),  
                dcc.Dropdown(id='msfile-selector', 
                             placeholder='Select files (.raw, .mzML, .d)', 
                             multi=True, style={'width': '100%'}, 
                             className="p-0 mt-1"),  
                dbc.Col(dbc.Button("Select all", id="select-all-msfiles",
                                   color="secondary", outline=True, n_clicks=0, 
                                   className="p-0 mt-1",
                                   style={"width": "80px", "height": "25px", 
                                          "textTransform": "none"}),
                        style={"text-align": "right"}),  
                html.Label("Identification report", style={"font-weight": "bold"}),
                dbc.Input(id='idreport-path', type='text', 
                          placeholder='Precursor-level identification report', 
                          style={'width': '100%'}, persistence=True),
                html.Label("Spectral library", style={"font-weight": "bold", "margin-top": "10px"}),
                dbc.Input(id='speclib-path', type='text', placeholder='Path of a spectral library file', 
                          style={'width': '100%'}, persistence=True),
                html.Label("Fasta files", style={"font-weight": "bold", "margin-top": "10px"}),
                dbc.Input(id='fasta-path', type='text', 
                          placeholder='Path of fasta files (.fasta, .fas)', 
                          style={'width': '100%'}, persistence=True),
                dcc.Dropdown(id='fasta-selector', placeholder='Select fasta files', 
                             multi=True, style={'width': '100%'}, 
                             className="p-0 mt-1"),
                dbc.Col(dbc.Button("Select all", id="select-all-fasta",
                                   color="secondary", outline=True, n_clicks=0, 
                                   className="p-0 mt-1",
                                   style={"width": "80px", "height": "25px", 
                                          "textTransform": "none"}),
                        style={"text-align": "right"}), 
                html.Label("Output directory", style={"font-weight": "bold"}),
                dbc.Input(id='outdir-path', type='text', placeholder='Path of output directory', 
                          style={'width': '100%'}, persistence=True),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Run", id="run-button", 
                                   color="primary",  
                                   style={"font-weight": "bold", "width": "120px"}),
                        html.Div(
                            dbc.Spinner(color="primary", size="sm", type="grow", id="run-spinner"),
                            id="spinner-container",
                            style={"display": "none", "marginLeft": "10px"}
                        )
                    ], width=7),
                    dbc.Col(
                        dbc.Button("Stop", id="stop-button", 
                                   color="danger", 
                                   style={"font-weight": "bold", "width": "120px"}),
                        width=5, style={"textAlign": "right"}
                    )
                ]),
                html.Div(id="stop-status", style={"display": "none"})
            ],
            vertical=True,
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Settings", id="param-button",
                               color="secondary", outline=True, 
                               n_clicks=0, className="mt-1",
                               style={'width': '120px', "textTransform": "none"}
                    )
                ], width=7),
                dbc.Col(
                    dbc.Button("View Results", id="view-prot", 
                               color="info", className="mt-1",
                               style={"width": "120px", "textTransform": "none"}),
                    width=5, style={"textAlign": "right"}
                )
            ]),
            dbc.Collapse(
                dbc.Card([
                    dbc.CardHeader(html.Label("Parameters", style={"font-weight": "bold"})),
                    dbc.CardBody([
                        html.P('General settings', style={"font-weight": "bold"}, className="mt-1 mb-0"),
                        dbc.Row([
                            dbc.Col(html.Label("log_level"), width=8),  
                            dbc.Col(dbc.Input(id="log-level", 
                                              type="number", 
                                              value=5, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("use_saved_rts"), width=8),  
                            dbc.Col(dbc.Checklist(id="use-saved-rts",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=[],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("use_saved_xic"), width=8),  
                            dbc.Col(dbc.Checklist(id="use-saved-xic",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=[],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("use_xic_report"), width=8),  
                            dbc.Col(dbc.Checklist(id="use-xic-report",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=[],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        html.P('ID results processing', style={"font-weight": "bold"}, className="mt-1 mb-0"),
                        dbc.Row([
                            dbc.Col(html.Label("min_runs_quant"), width=8),  
                            dbc.Col(dbc.Input(id="min-runs-quant", 
                                              type="number", 
                                              value=3, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("min_runs_model"), width=8),  
                            dbc.Col(dbc.Input(id="min-runs-model", 
                                              type="number", 
                                              value=3, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("min_runs_cal"), width=8),  
                            dbc.Col(dbc.Input(id="min-runs-cal", 
                                              type="number", 
                                              value=6, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        html.P('XIC extraction', style={"font-weight": "bold"}, className="mt-1 mb-0"),
                        dbc.Row([
                            dbc.Col(html.Label("mass_calibration"), width=8),  
                            dbc.Col(dbc.Checklist(id="mass-calibration",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=["yes"],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("fix_mass_acc"), width=8),  
                            dbc.Col(dbc.Checklist(id="fix-mass-acc",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=[],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("mass_acc_ms2"), width=8),  
                            dbc.Col(dbc.Input(id="mass-acc-ms2", 
                                              type="number", 
                                              value=15, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("mass_acc_ms1"), width=8),  
                            dbc.Col(dbc.Input(id="mass-acc-ms1", 
                                              type="number", 
                                              value=15, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("optimal_xic_source"), width=8),  
                            dbc.Col(dbc.Input(id="optimal-xic-source", 
                                              type="number", 
                                              value=0, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("int_removal"), width=8),  
                            dbc.Col(dbc.Checklist(id="int-removal",
                                                  options=[{"label": "Enabled", "value": "yes"}],
                                                  value=["yes"],
                                                  inline=True), 
                                    width=4, style={"text-align": "right"}),  
                        ], className="mt-0 mb-1"),
                        html.P('Normalization', style={"font-weight": "bold"}, className="mt-1 mb-0"),
                        dbc.Row([
                            dbc.Col(html.Label("norm_species"), width=8),  
                            dbc.Col(dbc.Input(id="norm-species", 
                                              type="text", 
                                              value='HUMAN', 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("norm_method"), width=8),  
                            dbc.Col(dbc.Input(id="norm-method", 
                                              type="number", 
                                              value=0, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                        dbc.Row([
                            dbc.Col(html.Label("trim_percentage"), width=8),  
                            dbc.Col(dbc.Input(id="trim-percentage", 
                                              type="number", 
                                              value=5, 
                                              style={"width": "80px", "height": "20px"}), 
                                    width=4),  
                        ], className="mt-0 mb-1"),
                    ])
                ], className="mt-1"),
                id="param-collapse",
                is_open=False,
            ),
        ]),
        html.Hr(),
    ],
    style=page_sidebar_style,
)

# Enhanced layout with smart scrolling
layout = html.Div(children=[
    # Store to track process and scroll state
    dcc.Store(id="process-state", data={"running": False, "start_time": None}),
    dcc.Store(id="scroll-state", data={"auto_scroll": True, "user_scrolled": False}),
                    
    # Interval for real-time log updates
    dcc.Interval(id="log-update-interval", interval=1000, n_intervals=0),
    html.Div(id="dummy-scroll-trigger", style={"display": "none"}),
    
    dbc.Row([
        dbc.Col(sidebar, width=4),
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Div(children=[
                # Log section with scroll controls
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span("Log", style={"flex": "1"}),
                            dbc.ButtonGroup([
                                dbc.Button(
                                    "Auto-scroll: ON", 
                                    id="auto-scroll-toggle",
                                    size="sm",
                                    color="success",
                                    outline=True,
                                    className="px-2 py-0 m-0",
                                    style={"height": "25px", "textTransform": "none"}
                                ),
                                dbc.Button(
                                    "↓ Bottom", 
                                    id="scroll-to-bottom",
                                    size="sm",
                                    color="info",
                                    outline=True,
                                    className="px-2 py-0 m-0",
                                    style={"height": "25px", "textTransform": "none"}
                                ),
                                dbc.Button(
                                    "↑ Top", 
                                    id="scroll-to-top",
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                    className="px-2 py-0 m-0",
                                    style={"height": "25px", "textTransform": "none"}
                                ),
                            ])
                        ], style={"display": "flex", "alignItems": "center", "width": "100%"})
                    ]),
                    dbc.CardBody([
                        dbc.Textarea(
                            id="log-output", 
                            style={
                                "width": "100%", 
                                "height": "500px",
                                "fontFamily": "'Consolas', 'Courier New', 'Fira Mono', monospace",
                                "fontSize": "13px"
                                }, 
                            readOnly=True)
                    ])
                ], style={"width": "97%"}),
                # Progress bar section
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span("Status:", style={"marginRight": "5px", "marginLeft": "5px"}),
                            html.Span(id="process-status", children="Ready")
                            ], style={"display": "flex", "alignItems": "center"})
                    ])
                ], body=False, style={"marginTop": "5px", "width": "97%"}),
            ]),
        ], width=8)
    ])
])


@app.callback(
    Output("param-collapse", "is_open"),
    [Input("param-button", "n_clicks")],
    [State("param-collapse", "is_open")],
)
def param_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# Global variables
current_process = None
log_queue = queue.Queue()
process_running = False
process_start_time = None

def run_jumpdiaq_process(config_path):
    global log_queue, current_process, process_running, process_start_time
    
    process_running = True
    process_start_time = time.time()
    
    # Non-blocking I/O approach
    process = subprocess.Popen(
        ['jumpdiaq', '--config-file', config_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Store the process reference
    current_process = process
    
    # Create threads to read stdout and stderr
    def read_stream(stream, prefix=""):
        for line in iter(stream.readline, ''):
            if line:
                log_queue.put(f"{prefix}{line.strip()}")
    
    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, ""), daemon=True)
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, ""), daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for process to complete
    process.wait()
    log_queue.put("[INFO] Process completed with return code: " + str(process.returncode))
    
    # Mark process as completed
    process_running = False
    current_process = None


# Callback to update log output
@app.callback(
    Output("log-output", "value"),
    Input("log-update-interval", "n_intervals"),
    State("log-output", "value"),
)
def update_log(n_intervals, current_logs):
    """Fetch logs from queue and update the display."""
    new_logs = []
    while not log_queue.empty():
        new_logs.append(log_queue.get())
    
    if new_logs:
        updated_logs = (current_logs or "") + "\n" + "\n".join(new_logs)
        return updated_logs
    return current_logs


# Smart auto-scroll callback - only scrolls when process is running AND auto-scroll is enabled
app.clientside_callback(
    """
    function(log_value, scroll_state, process_state) {
        console.log('Smart scroll triggered:', {
            'log_length': log_value ? log_value.length : 0,
            'auto_scroll': scroll_state ? scroll_state.auto_scroll : false,
            'process_running': process_state ? process_state.running : false
        });
        
        if (log_value && scroll_state && scroll_state.auto_scroll && process_state && process_state.running) {
            setTimeout(function() {
                var textarea = document.getElementById("log-output");
                if (textarea) {
                    textarea.scrollTop = textarea.scrollHeight;
                    console.log('Auto-scrolled to bottom');
                }
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-scroll-trigger", "children"),
    [
        Input("log-output", "value"),
        Input("scroll-state", "data"),
        Input("process-state", "data")
    ],
    prevent_initial_call=True
)

# Auto-scroll toggle button
@app.callback(
    [
        Output("auto-scroll-toggle", "children"),
        Output("auto-scroll-toggle", "color"),
        Output("scroll-state", "data", allow_duplicate=True)
    ],
    Input("auto-scroll-toggle", "n_clicks"),
    State("scroll-state", "data"),
    prevent_initial_call=True
)
def toggle_auto_scroll(n_clicks, scroll_state):
    """Toggle auto-scroll on/off."""
    if n_clicks:
        current_state = scroll_state.get("auto_scroll", True)
        new_state = not current_state
        
        button_text = "Auto-scroll: ON" if new_state else "Auto-scroll: OFF"
        button_color = "success" if new_state else "danger"
        
        new_scroll_state = {
            "auto_scroll": new_state,
            "user_scrolled": scroll_state.get("user_scrolled", False)
        }
        
        return button_text, button_color, new_scroll_state
    
    return dash.no_update, dash.no_update, dash.no_update

# Manual scroll to bottom button
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var textarea = document.getElementById("log-output");
            if (textarea) {
                textarea.scrollTop = textarea.scrollHeight;
                console.log('Manual scroll to bottom');
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-scroll-trigger", "children", allow_duplicate=True),
    Input("scroll-to-bottom", "n_clicks"),
    prevent_initial_call=True
)

# Manual scroll to top button
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            var textarea = document.getElementById("log-output");
            if (textarea) {
                textarea.scrollTop = 0;
                console.log('Manual scroll to top');
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-scroll-trigger", "children", allow_duplicate=True),
    Input("scroll-to-top", "n_clicks"),
    prevent_initial_call=True
)

# Detect user scrolling and disable auto-scroll if user scrolls up
app.clientside_callback(
    """
    function(n_intervals, scroll_state) {
        var textarea = document.getElementById("log-output");
        if (textarea && scroll_state && scroll_state.auto_scroll) {
            // Check if user has scrolled up from bottom
            var isAtBottom = textarea.scrollTop >= (textarea.scrollHeight - textarea.clientHeight - 10);
            
            if (!isAtBottom && !scroll_state.user_scrolled) {
                console.log('User scrolled up, disabling auto-scroll');
                return {
                    "auto_scroll": false,
                    "user_scrolled": true
                };
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("scroll-state", "data", allow_duplicate=True),
    [
        Input("log-update-interval", "n_intervals"),
        State("scroll-state", "data")
    ],
    prevent_initial_call=True
)


def format_elapsed_time(elapsed_time):
    if elapsed_time < 60:
        time_str = f"{int(elapsed_time)}s"
    else:
        mins = int(elapsed_time // 60)
        secs = int(elapsed_time % 60)
        time_str = f"{mins}m {secs}s"
    return time_str


# Progress bar updates
@app.callback(
    Output("process-status", "children"),
    Output("process-state", "data"),
    Input("log-output", "value"),
    Input("log-update-interval", "n_intervals"),
    State("process-state", "data"),
    prevent_initial_call=True
)
def update_process_state(log_value, n_intervals, process_state):
    """Update process state."""
    global process_running, process_start_time, current_process
    
    # Initialize process_state if None
    if process_state is None:
        process_state = {"running": False, "start_time": None}
    
    # Check if process is currently running
    if process_running and current_process and log_value:
        elapsed_time = time.time() - process_start_time if process_start_time else 0  
        time_str = format_elapsed_time(elapsed_time)
        
        status_text = html.Div([
            dbc.Spinner(
                color="primary",
                spinner_style={"width": "1.5rem", "height": "1.5rem", "marginRight": "8px"}
            ),
            html.Span(f"Running ... ({time_str} elapsed)")
        ], style={"display": "flex", "alignItems": "center"})
        
        new_process_state = {
            "running": True, 
            "start_time": process_start_time,
            "elapsed": elapsed_time
        }
        
        return status_text, new_process_state
    
    elif process_state.get("running", False) and not process_running:
        # Process just completed
        elapsed_time = process_state.get("elapsed", 0)
        time_str = format_elapsed_time(elapsed_time)
        
        # Check if process completed successfully or with errors
        if log_value.split('\n')[-1] == "[INFO] Process stopped":
            status_text = html.Div([
                html.I(className="fas fa-exclamation-triangle", 
                      style={"marginRight": "10px", "color": "red"}),
                "Stopped!"
            ])
        else:
            status_text = html.Div([
                html.I(className="fas fa-check-circle", 
                      style={"marginRight": "10px", "color": "green"}),
                f"Finished! (took {time_str})"
            ])
        
        new_process_state = {
            "running": False, 
            "start_time": None,
            "elapsed": elapsed_time,
        }
        
        return status_text, new_process_state
    
    else:
        # Process not running - return current state or defaults
        if not process_state.get("running", False) and not log_value:
            status_text = html.Div([
                html.I(className="fas fa-play-circle", 
                      style={"marginRight": "10px", "color": "gray"}),
                "Ready to start"
            ])
            return status_text, process_state
        
        return dash.no_update, dash.no_update


# Callback to load available CSV files from the directory
@app.callback(
    Output('stored-outdir', 'data'),
    Output("scroll-state", "data", allow_duplicate=True),
    Input('run-button', 'n_clicks'),
    State('msfile-path', 'value'), 
    State('msfile-selector', 'value'),
    State('idreport-path', 'value'),
    State('speclib-path', 'value'),
    State('fasta-path', 'value'), 
    State('fasta-selector', 'value'),
    State('outdir-path', 'value'),
    State('log-level', 'value'),
    State('use-saved-rts', 'value'),
    State('use-saved-xic', 'value'),
    State('use-xic-report', 'value'),
    State('min-runs-quant', 'value'),
    State('min-runs-model', 'value'),
    State('min-runs-cal', 'value'),
    State('mass-calibration', 'value'),
    State('fix-mass-acc', 'value'),
    State('mass-acc-ms2', 'value'),
    State('mass-acc-ms1', 'value'),
    State('optimal-xic-source', 'value'),
    State('int-removal', 'value'),
    State('norm-species', 'value'),
    State('norm-method', 'value'),
    State('trim-percentage', 'value'),
    prevent_initial_call=True
)
def run_jumpdiaq(
        n_clicks, 
        msfile_path,
        msfiles,
        idreport,
        speclib,
        fasta_path,
        fastas,
        outdir,
        log_level,
        use_saved_rts,
        use_saved_xic,
        use_xic_report,
        min_runs_quant,
        min_runs_model,
        min_runs_cal,
        mass_calibration,
        fix_mass_acc,
        mass_acc_ms2,
        mass_acc_ms1,
        optimal_xic_source,
        int_removal,
        norm_species,
        norm_method,
        trim_percentage
        ):
    if n_clicks == 0 or not outdir:
        return None, dash.no_update
    
    if not msfiles and outdir and os.path.isdir(outdir):
        logger.info(f"Quantification results will be read directly from {outdir}")
        return outdir, dash.no_update
    
    # Generate cmd
    cmd_args = []
    
    # Add MS files
    if msfiles and msfile_path:
        cmd_args.append("--input-runs")
        for msfile in msfiles:
            full_path = os.path.join(msfile_path, msfile)
            cmd_args.append(full_path)
    
    # Add report file
    if idreport:
        cmd_args.append("--id-report")
        cmd_args.append(str(Path(idreport)))
    
    # Add library file
    if speclib:
        cmd_args.append("--spec-lib ")
        cmd_args.append(str(Path(speclib)))
    
    if fastas and fasta_path:
        cmd_args.append("--fastas")
        for fasta in fastas:
            full_path = os.path.join(fasta_path, fasta)
            cmd_args.append(full_path)
    
    # Add output directory
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        outdir = str(outdir)
        cmd_args.append("--out-dir")
        cmd_args.append(outdir)
    
    # 
    cmd_args.append("--log-level")
    cmd_args.append(str(log_level))
    
    # 
    if 'yes' in use_saved_rts:
        cmd_args.append("--use-saved-rts")
        cmd_args.append("1")
        
    # 
    if 'yes' in use_saved_xic:
        cmd_args.append("--use-saved-xic")
        cmd_args.append("1")
            
    # 
    if 'yes' in use_xic_report:
        cmd_args.append("--use-xic-report")
        cmd_args.append("1")
    
    cmd_args.append("--min-runs-quant")
    cmd_args.append(str(min_runs_quant))
    cmd_args.append("--min-runs-model")
    cmd_args.append(str(min_runs_model))
    cmd_args.append("--min-runs-cal")
    cmd_args.append(str(min_runs_cal))
    
    if 'yes' not in mass_calibration:
        cmd_args.append("--mass-calibration")
        cmd_args.append("0")
        
    if 'yes' in fix_mass_acc:
        cmd_args.append("--fix-mass-acc")
        cmd_args.append("1")
        cmd_args.append("--mass-acc-ms2")
        cmd_args.append(str(mass_acc_ms2))
        cmd_args.append("--mass-acc-ms1")
        cmd_args.append(str(mass_acc_ms1))
        
    cmd_args.append("--optimal-xic-source")
    cmd_args.append(str(optimal_xic_source))
    
    # Add interference removal option
    if 'yes' not in int_removal:
        cmd_args.append("--int-removal") 
        cmd_args.append("0")
    
    cmd_args.append("--norm-species")
    cmd_args.append(str(norm_species))
    cmd_args.append("--norm-method")
    cmd_args.append(str(norm_method))
    cmd_args.append("--trim-percentage") 
    cmd_args.append(str(trim_percentage))
    
    #
    cmd_args.append("--show-progress")
    cmd_args.append("0")
    
    # Write config to file
    config_path = os.path.join(outdir, "jumpdiaq_config.txt")
    with open(config_path, 'w') as f:
        f.write('\n'.join(cmd_args))
    logger.info(f"Config file created at {config_path}")
    
    #
    run_jumpdiaq_process(config_path)
    
    return outdir, {"auto_scroll": True, "user_scrolled": False}


@app.callback(
    Output("spinner-container", "style"),
    Input("run-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State('msfile-selector', 'value'),
    State("process-state", "data"),
    prevent_initial_call=True
)
def running_indicator(run_clicks, stop_clicks, ms_files, process_state):
    if ctx.triggered_id == "run-button" and ms_files:
        return {"display": "inline-block", "marginLeft": "10px"}
    elif ctx.triggered_id == "stop-button" or process_state.get("running", False):
        return {"display": "none"}
    return dash.no_update


@app.callback(
    Output("stop-status", "children"),  # You can create a hidden div for this
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True
)
def stop_process(n_clicks):
    global current_process
    
    if current_process is not None:
        # Log that we're stopping the process
        log_queue.put("[INFO] Stopping the process...")
        
        try:
            # On Windows
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(current_process.pid)], 
                              check=False, capture_output=True)
            # On Linux/Mac
            else:
                current_process.terminate()
                # Give it some time to terminate gracefully
                try:
                    current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate in time
                    current_process.kill()
            
            log_queue.put("[INFO] Process stopped")
            current_process = None
            return "Process stopped"
            
        except Exception as e:
            log_queue.put(f"[ERROR] Failed to stop process: {str(e)}")
            return f"Failed to stop: {str(e)}"
    
    return "No process running"

# Callback: Update the msfile options when the directory path changes
@app.callback(
    Output('msfile-selector', 'options'),
    Input('msfile-path', 'value')  # Trigger when path changes
)
def list_ms_files(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return []

    ms_files = []
    
    # Find .raw files
    raw_files = [f for f in os.listdir(dir_path) if f.endswith('.raw')]
    ms_files.extend(raw_files)
    
    # Find .mzML files
    mzml_files = [f for f in os.listdir(dir_path) if f.endswith('.mzML')]
    ms_files.extend(mzml_files)

    # Find .d directories
    d_directories = [d for d in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, d)) and d.endswith('.d')]
    ms_files.extend(d_directories)

    return [{'label': f, 'value': f} for f in ms_files]

# Callback: Update the msfile options when the directory path changes
@app.callback(
    Output('fasta-selector', 'options'),
    Input('fasta-path', 'value')  # Trigger when path changes
)
def list_fasta_files(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return []

    # Find .fasta/.fas files
    fasta_files = [f for f in os.listdir(dir_path) if f.endswith('.fasta')]
    fas_files = [f for f in os.listdir(dir_path) if f.endswith('.fas')]
    fasta_files.extend(fas_files)

    return [{'label': f, 'value': f} for f in fasta_files]

@app.callback(
    Output('msfile-selector', 'value'),
    Input('select-all-msfiles', 'n_clicks'),
    State('msfile-selector', 'options'),
    State('msfile-selector', 'value')  # <- Add this back
)
def select_all_msfiles(n_clicks, options, current_value):
    if n_clicks == 0:
        return current_value

    if current_value and len(current_value) == len(options):
        return []

    return [option['value'] for option in options]

@app.callback(
    Output('fasta-selector', 'value'),
    Input('select-all-fasta', 'n_clicks'),
    State('fasta-selector', 'options'),
    State('fasta-selector', 'value')  # <- Add this back
)
def select_all_fastas(n_clicks, options, current_value):
    if n_clicks == 0:
        return current_value

    if current_value and len(current_value) == len(options):
        return []

    return [option['value'] for option in options]
