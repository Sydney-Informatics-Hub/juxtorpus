from unittest import TestCase

import sys
import re

import spacy
from spacy.matcher import Matcher
import pandas as pd

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.processors import process
from juxtorpus.corpus.processors.spacy_processor import *


def init_nlp_pymusas() -> spacy.Language:
    # We exclude the following components as we do not need them.
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    # Load the English PyMUSAS rule based tagger in a separate spaCy pipeline
    english_tagger_pipeline = spacy.load('en_dual_none_contextual')
    # Adds the English PyMUSAS rule based tagger to the main spaCy pipeline
    nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
    return nlp


class TestPymusas(TestCase):
    def test_Given_python_39_When_spacy_pymusas_pipeline_Then_processable(self):
        assert re.match(r'3\.9\.*', sys.version.split()[0]) is not None, "Expecting python version 3.9.x"
        try:
            nlp = init_nlp_pymusas()
            text = "The Nile is a major north-flowing river in Northeastern Africa."
            _ = nlp(text)
        except Exception as e:
            assert False, f"Running pymusas with python 3.9.x failed. Exception:\n{e}"

    def test_ltest_process_spacy_doc_with_multiple_pipelines(self): # -> can't use multiple pipelines
        nlps = [spacy.load(model) for model in ('en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf')]
        nlps.append(init_nlp_pymusas())

        nlp = spacy.blank('en')
        nlp.add_pipe('extract_hashtags')
        nlp.add_pipe('extract_mentions')
        nlps.append(nlp)

        from spacytextblob.spacytextblob import SpacyTextBlob
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')
        nlps.append(nlp)

        text = "The Nile is a major north-flowing river in Northeastern Africa."
        doc = text
        for nlp in nlps:
            print(nlp)
            doc = nlp(doc)

    def test_ltest_create_custom_dtm_with_pymusas_tags(self):
        nlp = init_nlp_pymusas()
        corpus = Corpus.from_dataframe(
            df=pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv', nrows=100),
            col_doc='processed_text',
        )
        scorpus: SpacyCorpus = process(corpus, nlp=nlp)

        cdtm = scorpus.create_custom_dtm(tokeniser_func=lambda doc: [tag for t in doc for tag in t._.pymusas_tags])
        # note: it incld. PUNCT tags we can ignore when creating the custom dtm.
        #  specifier/variants to a top level tag such as X2.5 and then X2.5+  where + is a specifier.



    def test_ltest_filter_corpus_by_pymusas_tags(self):
        nlp = init_nlp_pymusas()
        text = "The Nile is a major north-flowing river in Northeastern Africa."
        doc = nlp(text)
        matcher = Matcher(nlp.vocab)
        matcher.add('pymusas_Z5', patterns=[
            [{"_": {"pymusas_tags": {"INTERSECTS": ["Z5"]}}}]
        ])
        print()
        matches = matcher(doc)
        from pprint import pprint
        pprint(matches)



    # test with a spacy matcher on matching pymusas tags.
    # test - create custom dtm with pymusas tags
    # test - running multiple nlp pipelines on docs

    # workflow:
    #  1. process corpus with pymusas pipeline.
