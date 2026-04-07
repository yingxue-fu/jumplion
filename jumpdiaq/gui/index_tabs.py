# builtin
import logging
from importlib.metadata import version
# external
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
# local
from jumpdiaq.gui.app import app
from jumpdiaq.gui.pages import home, inputs, prots, precs, manual
from jumpdiaq.gui.utils import get_local_ip

# Sidebar Layout
sidebar = html.Div(
    [
        html.Div([
            html.Div([
                html.Img(src="/assets/logo.png", style={"width": "3rem"}),
                html.H5("JUMPlion", style={'color': '#40484E', 'margin-top': '10px'}),
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
                    html.I(className="fa-solid fa-book"),
                    html.Span("Manual", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-manual", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-gear"),
                    html.Span("Analyze", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-params", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-chart-line"),
                    html.Span("Proteins", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-prots", href="#", active="exact", className="pe-3"),
                dbc.NavLink([html.Div([
                    html.I(className="fa-solid fa-chart-area"),
                    html.Span("Precursors", style={'margin-top': '3px'})], className='icon_title')],
                    id="tab-precs", href="#", active="exact", className="pe-3"),
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
    #dcc.Store(id='stored-msfiles', storage_type='session'),
    dcc.Store(id='stored-outdir', storage_type='session'),
    dcc.Store(id='stored-meta-path', storage_type='session'),
    dcc.Store(id='stored-meta', storage_type='session'),
    dcc.Store(id='selected-prot', storage_type='session'),
    dcc.Store(id='stored-precursor-options', storage_type='session'),
    sidebar,
    html.Div([
        html.Div(id="page-home", children=home.layout, style={"display": "block"}),  # Default active tab
        html.Div(id="page-manual", children=manual.layout, style={"display": "none"}),
        html.Div(id="page-params", children=inputs.layout, style={"display": "none"}),
        html.Div(id="page-prots", children=prots.layout, style={"display": "none"}),
        html.Div(id="page-precs", children=precs.layout, style={"display": "none"}),
    ], id="page-content"),
])


# Callback to Switch Tabs Without Reloading
@app.callback(
    [Output("page-home", "style"),
     Output("page-manual", "style"),
     Output("page-params", "style"),
     Output("page-prots", "style"),
     Output("page-precs", "style")],
    [Input("tab-home", "n_clicks"),
     Input("tab-manual", "n_clicks"),
     Input("tab-params", "n_clicks"),
     Input("tab-prots", "n_clicks"),
     Input("tab-precs", "n_clicks"),
     Input("read-manual", "n_clicks"),
     Input("start-analysis", "n_clicks"),
     Input("start-analysis-manual", "n_clicks"),
     Input("view-prot", "n_clicks"),
     Input("view-prot-prec", "n_clicks"),
     Input("check-prec-quantity", "n_clicks"),
     Input("check-prec-quality", "n_clicks"),]
)
def switch_tabs(n1, n2, n3, n4, n5, read_click, start_click, start_click_manual, view_prot, view_prot_prec, check_prec_quant, check_prec_quality):
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
        {"display": "block"} if clicked in ["tab-manual", "read-manual"] else {"display": "none"},
        {"display": "block"} if clicked in ["tab-params", "start-analysis", "start-analysis-manual"] else {"display": "none"},
        {"display": "block"} if clicked in ["tab-prots", "view-prot", "view-prot-prec"] else {"display": "none"},
        {"display": "block"} if clicked in ["tab-precs", "check-prec-quantity", "check-prec-quality"] else {"display": "none"},
    ]


def main():
    logging.info(f"JUMPlion-GUI (v{version('jumpdiaq')})\n")
    local_ip = get_local_ip()
    app.run(host=local_ip, port=8050, debug=True)
    
