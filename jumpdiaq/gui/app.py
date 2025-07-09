# builtin
import logging
# external
import dash
import dash_bootstrap_components as dbc

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css"
app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, 
                                      dbc.icons.FONT_AWESOME,
                                      dbc_css]
                )

server = app.server

# Sidebar Style in a Page
page_sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": "10rem",
    "bottom": 0,
    "width": "35rem",
    "padding": "2rem 1rem",
    "overflowY": "auto",
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
