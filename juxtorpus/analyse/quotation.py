from quotation_tool.extract_display_quotes import QuotationTool

from juxtorpus.corpus import Corpus


def analyse_with_quotation(corpus: Corpus, doc_id: str,
                           add_results: bool, results_prefix='quotation',
                           entities: list[str] = None):
    if add_results:
        raise NotImplementedError("Unable to add results with quotation tool.")
    if entities is None:
        entities = ['ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'LOC']
    qt = QuotationTool.from_corpus(corpus=corpus, doc_name_meta_id=doc_id)
    return qt.get_quotes(entities), qt
