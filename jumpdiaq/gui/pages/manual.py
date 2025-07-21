# builtin
import re
import pathlib
# external
from dash import dcc, html
import dash_bootstrap_components as dbc
# local
from jumpdiaq.gui.app import page_sidebar_style


# Load markdown content
markdown_path = pathlib.Path(__file__).parent.parent.parent.parent / "README.md"
with open(markdown_path, 'r', encoding='utf-8') as f:
    markdown_text = f.read()

# Auto-generate TOC + anchor tags
def generate_toc_and_markdown(md_text):
    toc_links = []
    modified_lines = []

    for line in md_text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            anchor = re.sub(r"[^\w\- ]", "", title).strip().lower().replace(" ", "-")
            
            # TOC link (with indentation)
            toc_links.append(
                dbc.NavLink(title, href=f"#{anchor}", external_link=True, className=f"ms-{(level - 1) * 2}")
            )
            
            # Use HTML heading with id
            modified_lines.append(f'<h{level} id="{anchor}">{title}</h{level}>')
        else:
            modified_lines.append(line)

    return toc_links, "\n".join(modified_lines)


# Generate TOC + modified markdown
toc_links, modified_md = generate_toc_and_markdown(markdown_text)

sidebar = html.Div(
    [
        html.H3("Table of Contents"),
        html.Hr(),
        dbc.Nav(toc_links, vertical=True, pills=True),
        html.Hr(),
        dbc.Button("Start Analysis", id="start-analysis-manual", color="primary",
                   style={"textTransform": "none"}),
    ],
    style=page_sidebar_style,
)

layout = html.Div(children=[
    dbc.Row([
        # Sidebar TOC
        dbc.Col(sidebar, width=4),
        # Main Markdown content
        dbc.Col([
            dcc.Markdown(markdown_text, dangerously_allow_html=True, style={'lineHeight': '1.8'}),
        ], width=8, style={'marginTop': '40px'})
    ])
])
