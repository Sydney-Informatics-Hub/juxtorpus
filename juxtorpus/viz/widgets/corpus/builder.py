from typing import Callable
from ipywidgets import (SelectMultiple, Select, Layout, Box, HTML, Button, HBox, VBox,
                        Output, Text, Label, Dropdown, Checkbox)
import pandas as pd
import math

from juxtorpus.viz import Widget
from juxtorpus.viz.widgets import FileUploadWidget
from juxtorpus.corpus import generate_name
from juxtorpus.viz.style.ipyw import no_horizontal_scroll
from juxtorpus.utils.utils_ipywidgets import debounce

f_selector_layout = {'width': '98%', 'height': '100%'}
f_uploader_layout = {'width': '98%', 'height': '50px'}
box_df_layout = {'width': '100%', 'height': '100%'}


def format_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class CorpusBuilderWidget(Widget):
    def __init__(self, builder: 'CorpusBuilder'):
        self.builder = builder
        self._on_build_callback = lambda corpus: None

        self._corpus_name_placeholder = ''

    def widget(self):
        # left vbox
        top_labels_config = [('id', '100%'), ('document', '200px'), ('meta', '200px'), ('data type', '200px')]
        top_labels = HBox(
            list(map(lambda ls: HTML(f"<b>{ls[0]}</b>", layout=Layout(width=ls[1])), top_labels_config)),
            layout=Layout(display='flex')
        )

        checkbox_configs = {col: {'text': False, 'meta': True,
                                  'id_width': top_labels_config[0][1],
                                  'text_width': top_labels_config[1][1],
                                  'meta_width': top_labels_config[2][1],
                                  'dtype_width': top_labels_config[3][1]}
                            for col in self.builder.columns}
        panel = [top_labels] + [self._create_checkbox(col, config.get('text'), config.get('meta'), config)
                                for col, config in checkbox_configs.items()]

        # right vbox
        self._corpus_name_placeholder = generate_name()
        key_textbox = Text(description='Name:', placeholder=self._corpus_name_placeholder)
        key_textbox.layout = Layout(width='250px')
        key_textbox.style.description_width = '40px'  # just fits 'Name:'

        button = Button(description='Build')
        button.layout = Layout(width='150px')
        button_output = Output(layout=Layout(overflow='scroll hidden', height='20%'))

        def _on_click_key_textbox(event):
            self._corpus_name_placeholder = event.get('new')

        def _on_click_build_corpus(_):
            if button.description in ('Done', 'Retry'):
                button.description = 'Build'
                button.button_style = ''
                return
            for key, config in checkbox_configs.items():
                if config.get('text'):
                    self.builder.set_document_column(key)
                else:
                    if config.get('meta'):
                        dtype = config.get('dtype')
                        self.builder.add_metas(key, dtypes=dtype)
            button_output.clear_output()

            try:
                button.disabled = True
                button.description = "Building..."
                button.button_style = 'info'
                corpus = self.builder.build()
                corpus.name = self._corpus_name_placeholder
                if self._on_build_callback is not None:
                    self._on_build_callback(corpus)
                button.description = "Done"
                button.button_style = 'success'
                button.disabled = False
            except Exception as e:
                with button_output: print(f"Failed to build. {e}")
                import traceback
                traceback.print_exc()
                button.description = 'Retry'
                button.button_style = 'warning'
                button.disabled = False
                return

        key_textbox.observe(_on_click_key_textbox, names='value')
        button.on_click(_on_click_build_corpus)

        name_and_build_btn = HBox([key_textbox, button], layout=Layout(**no_horizontal_scroll))

        vertical_pad = Box(layout=Layout(width='100%', height='70%'))
        horizontal_pad = Box(layout=Layout(width='2%', height='100%'))

        return HBox([VBox(panel, layout=Layout(width='52%')),
                     horizontal_pad,
                     VBox([vertical_pad, name_and_build_btn, button_output], layout=Layout(width='48%'))],
                    layout=Layout(**no_horizontal_scroll))

    def _create_checkbox(self, id_: str, text_checked: bool, meta_checked: bool, config: dict):
        id_width, text_width, = config.get('id_width'), config.get('text_width')
        meta_width, dtype_width = config.get('meta_width'), config.get('dtype_width')

        label = Label(f"{id_}", layout=Layout(width=id_width))
        t_checkbox = Checkbox(value=text_checked, layout=Layout(width=text_width))
        t_checkbox.style.description_width = '5px'

        m_checkbox = Checkbox(value=meta_checked, layout=Layout(width=meta_width))
        m_checkbox.style.description_width = '5px'

        dtypes = sorted([k for k in self.WIDGET_DTYPES_MAP.keys()])
        dtype_dd = Dropdown(options=dtypes, value=dtypes[0], disabled=False,
                            layout=Layout(width=dtype_width))

        # dtype_dd.observe
        def _toggle_checkbox(event):
            if not event.get('new'):
                config['text'] = False
                config['meta'] = False
            else:
                if id(event.get('owner')) == id(t_checkbox):
                    m_checkbox.value = False
                    dtype_dd.value = dtypes[dtypes.index('text')]  # if ticked as text - sets dtype as str
                    config['text'] = True
                    config['meta'] = False
                elif id(event.get('owner')) == id(m_checkbox):
                    t_checkbox.value = False
                    dtype_dd.value = dtypes[dtypes.index('auto')]  # if ticked as text - sets dtype as str
                    config['meta'] = True
                    config['text'] = False

        def _update_dtype(event):
            config['dtype'] = self.WIDGET_DTYPES_MAP.get(event.get('new'))

        t_checkbox.observe(_toggle_checkbox, names='value')
        m_checkbox.observe(_toggle_checkbox, names='value')  # todo: toggle text and meta
        dtype_dd.observe(_update_dtype, names='value')
        return HBox([label, t_checkbox, m_checkbox, dtype_dd])

    WIDGET_DTYPES_MAP = {
        'auto': None,
        'decimal': 'float',
        'whole number': 'Int64',
        'text': 'str',
        'datetime': 'datetime',
        'category': 'category'
    }

    def set_callback(self, callback: Callable):
        self._on_build_callback = callback


class CorpusBuilderFileUploadWidget(Widget):
    """ CorpusBuilderWidget
    Workflow:
    1. upload file(s)
    2. invoke corpus builder widget.
    """

    DTYPES_MAP = {
        'auto': None,
        'decimal': 'float',
        'whole number': 'Int64',
        'text': 'str',
        'datetime': 'datetime',
        'category': 'category'
    }

    def __init__(self):
        self._files = dict()
        self._builder = None
        self._widget = self._create_file_uploader_entrypoint()

        # callbacks
        self._on_build_callback = lambda corpus, output: None

    def widget(self):
        return self._widget

    def _create_file_uploader_entrypoint(self):
        """ Creates a selectable FileUploaderWidget. This is the entrypoint. """
        f_selector = Select(layout=Layout(**f_selector_layout))
        fuw = FileUploadWidget()
        fuw._uploader.layout = Layout(**f_uploader_layout)

        def _callback_fileupload_to_fileselector(fuw, added):
            self._files = {p.name: {'path': p} for p in fuw.uploaded()}
            f_selector.options = [name for name in self._files.keys()]

        fuw.set_callback(_callback_fileupload_to_fileselector)

        box_file_stats = Box(layout=Layout(**box_df_layout))

        @debounce(0.3)
        def _observe_file_selected(event):
            # from pprint import pprint
            selected = event.get('new')
            for name, d in self._files.items():
                d['selected'] = True if name in selected else False

            # build files preview
            df_list = []
            for name, d in self._files.items():
                if d.get('selected'):
                    size = format_size(d.get('path').stat().st_size)
                    df_list.append((name, size))
            df = pd.DataFrame(df_list, columns=['name', 'size'])
            box_file_stats.children = (HTML(df.to_html(index=False, classes='table')),)

        f_selector.observe(_observe_file_selected, names='value')
        button_confirm = Button(description='Confirm',
                                layout=Layout(width='20%', height='50px'))
        hbox_uploader = HBox([VBox([f_selector, fuw.widget()], layout=Layout(width='50%', height='200px')),
                              VBox([box_file_stats, button_confirm], layout=Layout(width='50%', height='200px'))],
                             layout=Layout(width='100%', height='100%'))

        vbox = VBox([hbox_uploader, Box()])

        def on_click_confirm(_):
            if button_confirm.description in ('Done', 'Retry'):
                button_confirm.disabled = False
                button_confirm.description = 'Confirm'
                button_confirm.button_style = ''
                return
            try:
                button_confirm.disabled = True
                button_confirm.description = 'Loading...'
                button_confirm.button_style = 'info'
                from juxtorpus.corpus import CorpusBuilder
                selected_files = [d.get('path') for d in self._files.values() if d.get('selected')]
                if len(selected_files) <= 0: return
                builder = CorpusBuilder(selected_files)
                builder.set_callback(self._on_build_callback)
                vbox.children = (vbox.children[0], builder.widget())
                button_confirm.description = 'Done'
                button_confirm.button_style = 'success'
                button_confirm.disabled = False
            except Exception as e:
                print(e)
                button_confirm.description = 'Retry'
                button_confirm.button_style = 'warning'
                button_confirm.disabled = False

        button_confirm.on_click(on_click_confirm)
        return vbox

    def set_callback(self, callback: Callable):
        """ Callbacks that takes the built corpus as the argument. """
        self._on_build_callback = callback
