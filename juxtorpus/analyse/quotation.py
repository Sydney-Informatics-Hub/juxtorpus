from quotation_tool.extract_display_quotes import QuotationTool

from juxtorpus.corpus import Corpus


def analyse_with_quotation(corpus: Corpus, doc_id: str,
                           add_results: bool, results_prefix='quotation',
                           entities: list[str] = None):
    # note: add_results is unused here. Maybe used in the future.
    if entities is None:
        entities = ['ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'LOC']
    qt = QuotationTool.from_corpus(corpus=corpus, doc_name_meta_id=doc_id)
    quotes_df = qt.get_quotes(entities)
    keep_cols = ['text_name', 'quote', 'quote_entities', 'speaker', 'speaker_entities', 'verb', 'quote_type']
    quotes_df = quotes_df.loc[:, keep_cols]
    return Corpus.from_dataframe(quotes_df, col_doc='quote', name=results_prefix), qt
