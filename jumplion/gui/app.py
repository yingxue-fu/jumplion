# builtin
import logging
# external
import dash
import dash_bootstrap_components as dbc

# Light/dark Bootstrap themes offered by the top-bar theme switch, paired
# with the matching Plotly figure templates (loaded via load_figure_template
# in gui/precs.py).
THEME_LIGHT = dbc.themes.ZEPHYR
THEME_DARK = dbc.themes.DARKLY
PLOTLY_TEMPLATE_LIGHT = "vizro"
PLOTLY_TEMPLATE_DARK = "vizro_dark"

app = dash.Dash(__name__,
                title="JUMPlion Viewer",
                assets_folder="assets",
                suppress_callback_exceptions=True,
                external_stylesheets=[THEME_LIGHT,
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
        'format': 'svg',
        'filename': 'custom_plot',
        'height': 300,  # Adjust as needed
        'width': 250,  # Adjust as needed
        #'scale': 2      # Increase for higher resolution (for png/jpeg formats)
    }
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
