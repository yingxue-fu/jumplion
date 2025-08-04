# builtin
import re
import pathlib
# external
from dash import dcc, html
import dash_bootstrap_components as dbc
# local
# from jumpdiaq.gui.app import page_sidebar_style

page_sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": "5rem",
    "bottom": 0,
    "width": "18rem",
    "padding": "3rem 2rem 0rem 2rem", #  top, right, bottom, left
    "overflowY": "auto",
}

# Load markdown content
markdown_path = pathlib.Path(__file__).parent.parent / "assets" / "README.md"
with open(markdown_path, 'r', encoding='utf-8') as f:
    markdown_text = f.read()

def create_anchor_id(title):
    """Create consistent anchor IDs"""
    anchor = re.sub(r'[^\w\s-]', '', title.lower())
    anchor = re.sub(r'[-\s]+', '-', anchor)
    return anchor.strip('-')

def preprocess_markdown_with_anchors(md_text):
    """Add explicit anchor tags to headings and generate TOC"""
    lines = md_text.splitlines()
    processed_lines = []
    toc_links = []
    
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            anchor_id = create_anchor_id(title)
            
            # Add explicit anchor to the heading
            processed_lines.append(f'{match.group(1)} <a id="{anchor_id}"></a>{title}')
            
            # Add to TOC if level <= 3
            if level == 3:
                toc_links.append(
                    html.Li([
                        html.A(
                            title,
                            href=f"#{anchor_id}",
                            style={
                                'textDecoration': 'none',
                                'color': '#007bff',
                                'paddingLeft': f'{(level - 1) * 20}px'
                            }
                        )
                    ], style={'listStyle': 'none', 'marginBottom': '8px'})
                )
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines), toc_links

# Process markdown and generate TOC
processed_markdown, toc_links = preprocess_markdown_with_anchors(markdown_text)

sidebar = html.Div(
    [
        html.H3("Table of Contents"),
        html.Hr(),
        html.Ul(toc_links, style={'paddingLeft': '0'}),
        html.Hr(),
        dbc.Button("Start Analysis", id="start-analysis-manual", color="info",
                   style={"textTransform": "none"}),
    ],
    style=page_sidebar_style,
)

layout = html.Div([
    dbc.Row([
        # Sidebar TOC
        dbc.Col(sidebar, width=3),
        # Main Markdown content
        dbc.Col([
            dcc.Markdown(processed_markdown, dangerously_allow_html=True, className='markdown-container'),
        ], width=8, style={'marginTop': '50px'})
    ])
])