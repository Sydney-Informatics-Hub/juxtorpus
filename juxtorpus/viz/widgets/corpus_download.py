""" corpus_download.py

Provides the widget that allows you to download a corpus.
"""
import base64
import io

import ipywidgets as widgets

from juxtorpus.corpus import Corpus


def make_download_html_widget_for(corpus: Corpus, ext: str, **kwargs) -> widgets.HTML:
    match ext:
        case '.csv':
            buffer = io.BytesIO()
            corpus.to_csv(buffer)
            buffer.seek(0)
        case '.xlsx':
            buffer = io.BytesIO()
            corpus.to_excel(buffer)
            buffer.seek(0)
        case _:
            raise ValueError("ext must be .csv or .xlsx.")

    b64encoded: bytes = base64.b64encode(buffer.read())
    href = f"data:text/csv;base64,{b64encoded.decode()}"
    return widgets.HTML(f'<a href="{href}" download={corpus.name + ext}>Download<a/>', **kwargs)
    # note: somehow ipywidget's HTML does not direct href to start from /files but IPython's HTML does.
