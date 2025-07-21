# builtin
import logging
# external
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, 
                assets_folder="assets",
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.SANDSTONE, 
                                      dbc.icons.FONT_AWESOME]
                )

server = app.server

# Sidebar Style in a Page
page_sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": "5rem",
    "bottom": 0,
    "width": "24rem",
    "padding": "3rem 2rem 0rem 2rem", #  top, right, bottom, left
    "overflowY": "auto",
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
