from semantic_tagger import SemanticTagger
from tqdm.auto import tqdm
import re

from juxtorpus.corpus import Corpus


def analyse_with_sem_tagger(corpus: Corpus,
                            doc_id: str,
                            language: str,
                            mwe: bool = False,
                            add_results: bool = False, results_prefix='semtag', ):
    stagger_mwe = 'yes' if mwe else 'no'
    stagger = SemanticTagger()
    stagger.loading_semantic_tagger(language=language, mwe=stagger_mwe)
    stagger.mwe = stagger_mwe
    # todo: process upload - build a dataframe concatenating all the documents.
    text_df = corpus.to_dataframe().rename({corpus.COL_DOC: 'text', doc_id: 'text_name'}, axis=1)
    text_df = stagger.hash_gen(text_df)
    tqdm.pandas()
    text_df['text'] = text_df['text'].progress_apply(lambda text: re.sub(r'\n', '', text))
    stagger.text_df = text_df
    stagger.tag_text()
    return stagger
