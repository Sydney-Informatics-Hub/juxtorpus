""" corpus_download.py

Provides the widget that allows you to download a corpus.
"""
from pathlib import Path

from juxtorpus.corpus import Corpus
import ipywidgets as widgets
import html


def make_download_html_widget_for(corpus: Corpus, ext: str, **kwargs) -> widgets.HTML:
    match ext:
        case '.csv':
            path = f"{corpus.name}.csv"
            corpus.to_csv(path)
        case '.xlsx':
            path = f"{corpus.name}.xlsx"
            corpus.to_excel(path)
        case _:
            raise ValueError("ext must be .csv or .xlsx.")

    path = Path(path)
    href = html.escape("./" + path.name)
    default_fname = path.name
    return widgets.HTML(f'<a href=/files/{href} download={default_fname}>Download</a>', **kwargs)
    # note: somehow ipywidget's HTML does not direct href to start from /files but IPython's HTML does.
