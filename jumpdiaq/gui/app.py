# builtin
import logging
# external
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, 
                title="JUMPdiaq",
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

to_image_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'custom_plot',
        'height': 500,  # Adjust as needed
        'width': 1000,  # Adjust as needed
        'scale': 2      # Increase for higher resolution
    }
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
