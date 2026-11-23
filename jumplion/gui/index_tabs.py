# builtin
import logging
from importlib.metadata import version
# local
from jumplion.gui.app import app
from jumplion.gui import precs
from jumplion.gui.utils import get_local_ip

# Single-page app: only the Precursors page is wired in.
app.layout = precs.layout


def main():
    logging.info(f"JUMPlion Viewer (v{version('jumplion')})\n")
    local_ip = get_local_ip()
    app.run(host=local_ip, port=8050, debug=True)
