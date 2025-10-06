# builtin
from importlib.metadata import version
# external
from dash import html
import dash_bootstrap_components as dbc


layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("JUMPdiaq: Integrating MS1 and MS2 Signals for Complete and Accurate Quantification in DIA Proteomics", 
                                className="card-title"),
                        html.Div([
                            html.Span(f"Version: {version('jumpdiaq')}", style={"marginRight": "20px"}),
                            html.Span([
                                "Github: ",
                                html.A("https://github.com/yingxue-fu/jumpdiaq", 
                                       href="https://github.com/yingxue-fu/jumpdiaq", 
                                       target="_blank")
                            ])
                        ], style={"display": "flex", "alignItems": "center"}),
                        dbc.Button("Read Manual", id="read-manual", color="info",
                                   style={"textTransform": "none"}, 
                                   className="me-3 mt-1"),
                        dbc.Button("Start Analysis", id="start-analysis", color="primary",
                                   style={"textTransform": "none"}, 
                                   className="me-3 mt-1"),
                        dbc.Carousel(
                            items=[
                                {
                                    "key": "1",
                                    "src": "/assets/images/prot1.png",
                                    "header": "",
                                    "caption": "",
                                },
                                {
                                    "key": "2",
                                    "src": "/assets/images/prot2.png",
                                    "header": "",
                                    "caption": "",
                                },
                                {
                                    "key": "3",
                                    "src": "/assets/images/prot3.png",
                                    "header": "",
                                    "caption": "",
                                },
                                {
                                    "key": "4",
                                    "src": "/assets/images/prot4.png",
                                    "header": "",
                                    "caption": "",
                                },
                            ],
                        variant="dark",
                        interval=2000,
                        class_name="carousel-fade",
                        style={"width": "85%"}),
                    ]
                ),
            ),
        ], width=12),
    ], style={"marginTop": "2rem"})
])

