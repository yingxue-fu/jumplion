# builtin
import os
import platform
from pathlib import Path
import subprocess
import queue
import threading
# external
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
# local
from jumpdiaq.gui.app import app, page_sidebar_style, logger


# app.clientside_callback(
#     """
#     function(children) {
#         var textarea = document.getElementById("log-output");
#         if (textarea) {
#             textarea.scrollTop = textarea.scrollHeight;  // Scroll to the bottom
#         }
#         return null;
#     }
#     """,
#     Output("dummy-scroll-trigger", "children", allow_duplicate=True),
#     Input("dummy-scroll-trigger", "children"),
#     prevent_initial_call=True
# )



sidebar = html.Div(
    [
        html.H3("Input"),
        #html.Hr(),
        dbc.Nav(
            [   
                html.Label("MS Files", style={"font-weight": "bold"}),
                #dbc.Row([
                dcc.Input(id='msfile-path', type='text', 
                                  placeholder='Folder of MS files (.mzML, .d)', 
                                  style={'width': '100%'}, persistence=True),#, width=9),  
                    # dbc.Col(dbc.Button("load", id="msfile-button", className="mb-3",
                    #                    color="secondary", outline=True, n_clicks=0),
                    #         className="d-flex justify-content-end"),  
                #], className="mb-2"),
                #dcc.Dropdown(id='msfile-selector', placeholder='Select files', multi=True),
                #dbc.Row([
                    #dbc.Col(
                dcc.Dropdown(id='msfile-selector', placeholder='Select files', multi=True), #width=8),
                dbc.Col(
                    dbc.Button("Select All", id="select-all-button", className="mb-3",
                                       color="secondary", outline=True, n_clicks=0, style={"width": "100px"}),
                                       className="d-flex justify-content-end"),
                #]), 
                html.Br(),
                html.Label("Identification Report", style={"font-weight": "bold"}),
                dcc.Input(id='idreport-path', type='text', placeholder='Precursor-level identification', 
                          style={'width': '100%'}, persistence=True),
                html.Br(),
                html.Label("Spectral Library", style={"font-weight": "bold"}),
                dcc.Input(id='speclib-path', type='text', placeholder='Path of a spectral library file', 
                          style={'width': '100%'}, persistence=True),
                html.Br(),
                html.Label("Output Directory", style={"font-weight": "bold"}),
                dcc.Input(id='outdir-path', type='text', placeholder='Path of output directory', 
                          style={'width': '100%'}, persistence=True),
                html.Br(),
                html.Label("Meta File Path", style={"font-weight": "bold"}),
                dcc.Input(id='meta-path', type='text', placeholder='Path of sample meta info', 
                          style={'width': '100%'}, persistence=True),
                html.Br(),
                #html.Button('Run', id='run-button', n_clicks=0, style={'margin-top': '10px'}),
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Button("Run", id="run-button", color="primary", className="mb-3", style={"width": "100px"}),
                            dbc.Button("Stop", id="stop-button", color="danger", className="mb-3 ms-4", style={"width": "100px"})
                        ], className="d-flex justify-content-end"),
                        width=12
                    )
                ]),
                html.Div(id="stop-status", style={"display": "none"}),
                html.Br(),
            ],
            vertical=True,
            pills=True,
        ),
        #html.Hr(),
        html.Div(
            [
                dbc.Button(
                    "Set Parameters",
                    id="param-button",
                    className="mb-3",
                    color="secondary",
                    outline=True, 
                    n_clicks=0,
                ),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader(html.Label("Parameters", style={"font-weight": "bold"})),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(html.Label("MS1 accuracy (ppm):"), width=8),  
                                dbc.Col(dcc.Input(id="ms1-acc", type="number", value=10, style={"width": "80px"}), width=4),  
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(html.Label("MS2 accuracy (ppm):"), width=8),  
                                dbc.Col(dcc.Input(id="ms2-acc", type="number", value=10, style={"width": "80px"}), width=4),  
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(html.Label("IM window:"), width=8),  
                                dbc.Col(dcc.Input(id="im-window", type="number", value=0.01, style={"width": "80px"}), width=4),  
                            ], className="mb-2"),                            
                            dbc.Row([
                                dbc.Col(html.Label("Interference removal:"), width=8),  
                                dbc.Col(dbc.Checklist(
                                    id="int-removal",
                                    options=[{"label": "Enabled", "value": "yes"}],
                                    value=["yes"],
                                    inline=True,  # Ensures the checkbox and label are in one line
                                ), width=4, style={"text-align": "right"}),  # Aligns to the right
                            ], className="mb-2"),
                        ])
                    ], className="mx-auto"),
                    id="param-collapse",
                    is_open=False,
                ),
            ]
        )
    ],
    style=page_sidebar_style,
)


layout = html.Div(children=[
    dbc.Row([
        dbc.Col(sidebar, width=3, md={"size": 3, "offset": 1}),
        dbc.Col([
            html.Br(),
            html.Div(children=[
                dbc.Card([
                    dbc.CardHeader("Log"),
                    dbc.CardBody([
                        dcc.Textarea(
                            id="log-output", 
                            style={
                                "width": "100%", 
                                "height": "600px",
                                "fontFamily": "'Consolas', 'Courier New', 'Fira Mono', monospace",
                                "fontSize": "14px"
                                }, 
                            readOnly=True)
                    ])
                ]),
                # Interval for real-time log updates
                dcc.Interval(id="log-update-interval", interval=1000, n_intervals=0),
                #html.Div(id="dummy-scroll-trigger", style={"display": "none"}),
            ]),
        ], width=8, className="mx-auto")
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


# Global variable to store the current process
current_process = None
log_queue = queue.Queue()  # Queue to store log messages
def run_diaq_process(cmd_args):
    """Runs main.py in a separate thread."""
    global log_queue, current_process
    
    main_py = str(Path("../main.py").resolve())
    cmd = ["python", main_py] + cmd_args
    
    # Non-blocking I/O approach
    process = subprocess.Popen(
        cmd, 
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


# Callback to update log output, without triggering scroll behavior
@app.callback(
    Output("log-output", "value"),
    Input("log-update-interval", "n_intervals"),
    State("log-output", "value"),
)
def update_log(n_intervals, current_logs):
    """Fetch logs from queue and update the display without scroll."""
    new_logs = []
    while not log_queue.empty():
        new_logs.append(log_queue.get())
    
    if new_logs:
        return (current_logs or "") + "\n" + "\n".join(new_logs)
    return current_logs  # No new logs, keep existing logs


# # Separate callback for auto-scrolling behavior, only when new logs are added
# @app.callback(
#     Output("dummy-scroll-trigger", "children"),
#     Input("log-output", "value"),  # Trigger on log change
# )
# def auto_scroll_on_new_log(log_value):
#     """Trigger scroll behavior only when new logs are added."""
#     if log_value:
#         return time.time()  # Use a timestamp to force the scroll when new logs are added
#     return dash.no_update


# Callback to load available CSV files from the directory
@app.callback(
    Output('stored-outdir', 'data'),
    Output('stored-meta-path', 'data'),
    Input('run-button', 'n_clicks'),
    State('msfile-path', 'value'), 
    State('msfile-selector', 'value'),
    State('idreport-path', 'value'),
    State('speclib-path', 'value'),
    State('outdir-path', 'value'),
    State('meta-path', 'value'),
    State('ms1-acc', 'value'),
    State('ms2-acc', 'value'),
    State('im-window', 'value'),
    State('int-removal', 'value'),
)
def run_diaq(
        n_clicks, 
        msfile_path,
        msfiles,
        idreport,
        speclib,
        outdir,
        meta_path,
        ms1_acc,
        ms2_acc,
        im_window,
        int_removal,
        ):
    if n_clicks == 0 or not outdir or not meta_path:
        return None, None
    
    if not msfiles and outdir and meta_path and os.path.isdir(outdir):
        logger.info(f"Quantification results will be read directly from {outdir}")
        return outdir, meta_path
    
    # Generate config file
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
        cmd_args.append("--spec-lib")
        cmd_args.append(str(Path(speclib)))
    
    # Add output directory
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        outdir = str(outdir)
        cmd_args.append("--out-dir")
        cmd_args.append(outdir)
    
    if meta_path:
        meta_path = Path(meta_path)
    
    # Add MS1 accuracy
    if ms1_acc:
        cmd_args.append("--mass-acc-ms1")
        cmd_args.append(str(ms1_acc))
    
    # Add MS2 accuracy
    if ms2_acc:
        cmd_args.append("--mass-acc-ms2")
        cmd_args.append(str(ms2_acc))
    
    # Add IM window if specified
    if im_window:
        cmd_args.append("--im-window")
        cmd_args.append(str(im_window))
    
    # Add interference removal option
    if int_removal == 'No':
        cmd_args.append("--int-removal")
        cmd_args.append("0") 
    
    # Write config to file
    config_path = os.path.join(outdir, "jumpdiaq_config.txt")
    with open(config_path, 'w') as f:
        f.write('\n'.join(cmd_args))
    
    logger.info(f"Config file created at {config_path}")
    
    #
    run_diaq_process(cmd_args)
    
    return outdir, meta_path


# Callback: Update the msfile options when the directory path changes
@app.callback(
    Output('msfile-selector', 'options'),
    Input('msfile-path', 'value')  # Trigger when path changes
)
def list_ms_files(dir_path):
    if not dir_path or not os.path.isdir(dir_path):
        return []

    ms_files = []

    # Find .mzML files
    mzml_files = [f for f in os.listdir(dir_path) if f.endswith('.mzML')]
    ms_files.extend(mzml_files)

    # Find .d directories
    d_directories = [d for d in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, d)) and d.endswith('.d')]
    ms_files.extend(d_directories)

    return [{'label': f, 'value': f} for f in ms_files]

@app.callback(
    Output('msfile-selector', 'value'),
    Input('select-all-button', 'n_clicks'),
    State('msfile-selector', 'options'),
    State('msfile-selector', 'value')  # <- Add this back
)
def select_all_files(n_clicks, options, current_value):
    if n_clicks == 0:
        return current_value

    if current_value and len(current_value) == len(options):
        return []

    return [option['value'] for option in options]



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