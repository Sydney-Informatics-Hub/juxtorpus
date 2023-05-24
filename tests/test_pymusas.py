from unittest import TestCase

import sys
import re

import spacy


class TestPymusas(TestCase):
    def test_Given_python_39_When_spacy_pymusas_pipeline_Then_processable(self):
        assert re.match(r'3\.9\.*', sys.version.split()[0]) is not None, "Expecting python version 3.9.x"
        try:
            # We exclude the following components as we do not need them.
            nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
            # Load the English PyMUSAS rule based tagger in a separate spaCy pipeline
            english_tagger_pipeline = spacy.load('en_dual_none_contextual')
            # Adds the English PyMUSAS rule based tagger to the main spaCy pipeline
            nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)

            text = "The Nile is a major north-flowing river in Northeastern Africa."
            _ = nlp(text)
        except Exception as e:
            assert False, f"Running pymusas with python 3.9.x failed. Exception:\n{e}"
