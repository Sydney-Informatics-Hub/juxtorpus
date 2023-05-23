from quotation_tool.extract_display_quotes import QuotationTool

from juxtorpus.corpus import Corpus


def analyse_with_quotation(corpus: Corpus, doc_id: str,
                           add_results: bool=False, results_prefix='quotation_',
                           entities: list[str] = None):
    # note: add_results is unused here. Maybe used in the future.
    if entities is None:
        entities = ['ORG', 'PERSON', 'GPE', 'NORP', 'FAC', 'LOC']
    qt = QuotationTool.from_corpus(corpus=corpus, doc_name_meta_id=doc_id)
    quotes_df = qt.get_quotes(entities)
    keep_cols = ['text_name', 'quote', 'quote_entities', 'speaker', 'speaker_entities', 'verb', 'quote_type']
    quotes_df = quotes_df.loc[:, keep_cols].astype(str)

    quotes_df['quote_entities'] = quotes_df['quote_entities'].apply(lambda t: t.replace('[', '').replace(']', ''))
    quotes_df['speaker_entities'] = quotes_df['speaker_entities'].apply(lambda t: t.replace('[', '').replace(']', ''))
    quotes_df['quote_type'] = quotes_df['quote_type'].astype('category')

    meta_cols = corpus._meta_registry.keys()
    corpus_meta = corpus.to_dataframe().loc[meta_cols]

    quotes_df.merge(corpus_meta, how='left', on='text_name')

    return Corpus.from_dataframe(quotes_df, col_doc='quote', name=corpus.name + results_prefix), qt
