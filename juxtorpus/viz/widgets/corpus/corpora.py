from typing import Optional
from abc import ABC

from ipywidgets import Label, Layout, HBox, GridBox, VBox, Button, HTML, Box, ButtonStyle
from ipywidgets import Checkbox
from juxtorpus.viz.style.ipyw import (
    center_style, corpus_id_layout, size_layout, parent_layout, download_layout, hbox_style
)
from juxtorpus.viz import Widget
from juxtorpus.viz.widgets.corpus.slicer import SlicerWidget
from juxtorpus.viz.widgets.corpus.builder import CorpusBuilderFileUploadWidget
from juxtorpus.viz.widgets.corpus_download import make_download_html_widget_for

import logging

logger = logging.getLogger()

_SHOW_BUILDER_DESCRIPTION = 'Upload a Corpus'
_HIDE_BUILDER_DESCRIPTION = 'Hide'


class CorporaWidget(Widget, ABC):
    """ CorporaWidget
    This class holds all the logic associated with the corpora widget.
    + Selectable corpus registry
    + Once selected, the Corpus Slicer panel will pop down.
    """

    _corpus_selector_labels = [
        HTML("<b>Corpus</b>", layout=Layout(**corpus_id_layout, **center_style)),
        HTML("<b>Size</b>", layout=Layout(**size_layout, **center_style)),
        HTML("<b>Parent</b>", layout=Layout(**parent_layout, **center_style)),
        HTML("<b>Download</b>", layout=Layout(**parent_layout, **center_style))
    ]

    def __init__(self, corpora: 'Corpora'):
        self.corpora = corpora

        self._builder: CorpusBuilderFileUploadWidget = CorpusBuilderFileUploadWidget()
        self._builder.set_callback(self._on_build_add_to_self)

        self._selector: VBox = self._corpus_selector()
        self._selected_checkbox = None

        self._widget = VBox([self._toggle_builder_button(), self._create_empty(), self._selector],
                            layout=Layout(grid_template_columns='repeat(2, 1fr)'))

    def widget(self) -> GridBox:
        self._refresh_corpus_selector()
        return self._widget

    def _corpus_selector(self, selected: Optional[str] = None) -> VBox:
        """ Creates the header and a row corresponding to each corpus in the corpora. """
        hbox_labels = HBox(self._corpus_selector_labels, layout=Layout(**hbox_style))
        rows = [self._corpus_selector_row(name) for name in self.corpora.items()]
        if selected:
            for r in rows:
                checkbox = r.children[0]
                checkbox.value = checkbox.description == selected
        return VBox([hbox_labels] + rows)

    def _corpus_selector_row(self, name) -> HBox:
        """ Creates a corpus row. """
        checkbox = Checkbox(description=f"{name}", layout=Layout(**corpus_id_layout))
        checkbox.style = {'description_width': '0px'}
        checkbox.observe(self._observe_row_checkbox, names='value')

        corpus = self.corpora.get(name)
        if not corpus:
            raise RuntimeError(f"Corpus: {name} does not exist. This should not happen.")

        parent_label = self._parent_label_of(corpus)
        checkbox.add_class('corpus_id_focus_colour')  # todo: add this HTML to code
        gen_dl_link_btn = Button(description="Generate Link")
        btn_style = ButtonStyle()
        btn_style.button_color = 'lightblue'
        btn_style.width = '60px'
        btn_style.height = '15px'
        btn_style.font_size = '6px'
        gen_dl_link_btn.style = btn_style

        download_hbox = HBox([
            gen_dl_link_btn,
            Box()
        ], layout=Layout(**download_layout))

        def on_generate_download_link(_):
            download_hbox.children = (
                gen_dl_link_btn,
                make_download_html_widget_for(corpus, ext='.xlsx')
            )

        gen_dl_link_btn.on_click(on_generate_download_link)

        hbox = HBox([checkbox,
                     Label(str(len(corpus)), layout=Layout(**size_layout)),
                     Label(parent_label, layout=Layout(**parent_layout)),
                     download_hbox,
                     ],
                    layout=Layout(**hbox_style))

        return hbox

    def _observe_row_checkbox(self, event):
        value, owner = event.get('new'), event.get('owner')
        if value:
            selected = self.corpora.get(owner.description.strip())
            self._selected_checkbox = owner
            if not selected:
                raise RuntimeError(f"Corpus: {owner.description} does not exist. This should not happen.")
            self._toggle_checkboxes(owner)

            slicer_widget = SlicerWidget(selected)
            slicer_widget._ops_widget.set_callback(self._on_slice_add_to_self)
            if self._slicer_appeared():
                self._widget.children = (*self._widget.children[:3], slicer_widget.widget())
            else:
                self._widget.children = (*self._widget.children, slicer_widget.widget())
        else:
            self._widget.children = (*self._widget.children[:3],)

    def _toggle_checkboxes(self, checked: Checkbox):
        if checked is None: return
        for hboxes in self._selector.children:
            for cb in hboxes.children:
                if isinstance(cb, Checkbox):
                    cb.value = cb.description == checked.description

    def _toggle_builder_button(self):
        button = Button(description=_SHOW_BUILDER_DESCRIPTION,
                        layout=Layout(width='300px'))

        def _on_click_toggle(_):
            if self._builder_appeared():
                self._widget.children = (self._widget.children[0], self._create_empty(), *self._widget.children[2:])
                button.description = _SHOW_BUILDER_DESCRIPTION
            else:
                self._widget.children = (self._widget.children[0], self._builder.widget(), *self._widget.children[2:])
                button.description = _HIDE_BUILDER_DESCRIPTION

        button.on_click(_on_click_toggle)
        return button

    def _on_slice_add_to_self(self, subcorpus):
        """ Add subcorpus to self on slice. """
        self.corpora.add(subcorpus)
        self._refresh_corpus_selector()

    def _on_build_add_to_self(self, corpus):
        self.corpora.add(corpus)
        self._refresh_corpus_selector()

    def _refresh_corpus_selector(self):
        self._selector = self._corpus_selector()
        self._toggle_checkboxes(self._selected_checkbox)
        if self._slicer_appeared():
            self._widget.children = (*self._widget.children[:2], self._selector, *self._widget.children[3:])
        else:
            self._widget.children = (*self._widget.children[:2], self._selector,)

    def _builder_appeared(self):
        return not self._is_empty(self._widget.children[1])

    def _slicer_appeared(self):
        return len(self._widget.children) > 3

    def _create_empty(self) -> Label:
        return Label(layout=Layout(height='0px', width='0px'))

    def _is_empty(self, widget) -> bool:
        return type(widget) == Label and widget.layout.height == '0px' and widget.layout.width == '0px'

    @staticmethod
    def _parent_label_of(corpus) -> str:
        return corpus.parent.name if corpus.parent else ''
