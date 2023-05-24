from semantic_tagger import SemanticTagger

from juxtorpus.corpus import Corpus

def analyse_with_sem_tagger(corpus: Corpus,
                            doc_id: str,
                            language: str,
                            mwe: bool = False,
                            add_results: bool = False, results_prefix='semtag',):
    stagger = SemanticTagger()
    stagger.loading_semantic_tagger(language=language, mwe='yes' if mwe else 'no')
    # todo: process upload - build a dataframe concatenating all the documents.
    col_text = corpus.COL_DOC
    col_text_name = doc_id

