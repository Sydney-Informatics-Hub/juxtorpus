""" corpus_download.py

Provides the widget that allows you to download a corpus.
"""

from juxtorpus.corpus import Corpus
from IPython.display import FileLink
import ipywidgets as widgets
import os


class DownloadFileLink(FileLink):
    '''
    Create link to download files in Jupyter Notebook
    '''
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape

        fp = "".join([self.url_prefix, escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str.format(
                    link=fp, file_name=self.file_name, link_text=self.link_text
                ),
                self.result_html_suffix,
            ]
        )


def make_download_btn(corpus: Corpus, ext: str):
    if ext not in ('.csv', '.xlsx'):
        raise ValueError("ext must be .csv or .xlsx.")
    match ext:
        case '.csv':
            path = f"{corpus.name}.csv"
            corpus.to_csv(path)
        case '.xlsx':
            path = f"{corpus.name}.xlsx"
            corpus.to_excel(path)
    return DownloadFileLink(path)
