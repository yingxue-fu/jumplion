# external
from dash import html
import dash_bootstrap_components as dbc


layout = html.Div([
    dbc.Card(
        dbc.CardBody(
            [
                html.H4("JUMPdiaq: a tool for DIA-MS data quantification", className="card-title"),
                html.P("write something"),
                dbc.Button("Start Analysis", id="start-analysis", color="primary"),
            ]
        ),
        style={
            "marginTop": "5rem",   
            "marginLeft": "10rem",
            "marginRight": "5rem"}  # Add left margin here
    )
])

