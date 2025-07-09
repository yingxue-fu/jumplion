# external
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
# local
from app import app
from pages import home, inputs, prots, precs, about


# Sidebar Layout
sidebar = html.Div(
    [
        html.Div([
            html.Div([
                html.Img(src="/assets/logo.png", style={"width": "5rem"}),
                html.H5("JUMPdiaq", style={'color': '#40484E', 'margin-top': '10px'}),
            ], className='image_title')
        ], className="sidebar-header"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-house"),
                    html.Span("Home", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-home", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-gear"),
                    html.Span("Parameters", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-inputs", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-chart-line"),
                    html.Span("Proteins", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-prots", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-chart-area"),
                    html.Span("Precursors", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-precs", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-circle-info"),
                    html.Span("About", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-about", href="#", active="exact", className="pe-3"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
    ],
    id="bg_id",
    className="sidebar",
)


# Main Layout with All Pages Preloaded but Hidden
app.layout = html.Div([
    dcc.Store(id='stored-msfiles', storage_type='session'),
    dcc.Store(id='stored-outdir', storage_type='session'),
    dcc.Store(id='stored-meta-path', storage_type='session'),
    dcc.Store(id='stored-meta', storage_type='session'),
    dcc.Store(id='selected-prot', storage_type='session'),
    dcc.Store(id='stored-precursor-options', storage_type='session'),
    sidebar,
    html.Div([
        html.Div(id="page-home", children=home.layout, style={"display": "block"}),  # Default active tab
        html.Div(id="page-inputs", children=inputs.layout, style={"display": "none"}),
        html.Div(id="page-prots", children=prots.layout, style={"display": "none"}),
        html.Div(id="page-precs", children=precs.layout, style={"display": "none"}),
        html.Div(id="page-about", children=about.layout, style={"display": "none"}),
    ], id="page-content"),
])


# Callback to Switch Tabs Without Reloading
@app.callback(
    [Output("page-home", "style"),
     Output("page-inputs", "style"),
     Output("page-prots", "style"),
     Output("page-precs", "style"),
     Output("page-about", "style")],
    [Input("tab-home", "n_clicks"),
     Input("tab-inputs", "n_clicks"),
     Input("tab-prots", "n_clicks"),
     Input("tab-precs", "n_clicks"),
     Input("tab-about", "n_clicks"),
     Input("start-analysis", "n_clicks")]
)
def switch_tabs(n1, n2, n3, n4, n5, start_click):
    ctx = callback_context
    if not ctx.triggered:
        return [{"display": "block"}, 
                {"display": "none"}, 
                {"display": "none"}, 
                {"display": "none"}, 
                {"display": "none"}]
    
    clicked = ctx.triggered[0]["prop_id"].split(".")[0]

    return [
        {"display": "block"} if clicked == "tab-home" else {"display": "none"},
        {"display": "block"} if clicked in ["tab-inputs", "start-analysis"] else {"display": "none"},
        {"display": "block"} if clicked == "tab-prots" else {"display": "none"},
        {"display": "block"} if clicked == "tab-precs" else {"display": "none"},
        {"display": "block"} if clicked == "tab-about" else {"display": "none"},
    ]


# if __name__ == '__main__':
def main():
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    app.run(host=host_ip, port=8050, debug=True)
