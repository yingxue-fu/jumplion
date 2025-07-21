# external
from dash import html
import dash_bootstrap_components as dbc


layout = html.Div([
    dbc.Card(
        dbc.CardBody(
            [
                html.H4("JUMPdiaq: a tool for DIA-MS data quantification", className="card-title"),
                html.P("Schematic of JUMPdiaq workflow"),
                html.P("[A figure]"),
                dbc.Button("Read Manual", id="read-manual", color="info",
                           style={"textTransform": "none"}, className="me-3"),
                dbc.Button("Start Analysis", id="start-analysis", color="primary",
                           style={"textTransform": "none"}),
            ]
        ),
        style={
            "marginTop": "3rem",   
            "marginLeft": "8rem",
            "marginRight": "3rem"}  # Add left margin here
    )
])

