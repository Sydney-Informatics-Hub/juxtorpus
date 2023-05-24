""" Linguistic matchers

1. compound nouns, verbs, adjectives; high school student, pick up, open minded,
3. noun phrase
4. verb phrase
4.
"""

NOUN = {'POS': 'NOUN'}
COMPOUND_NOUN = {**NOUN, 'POS': '+'}
ADJ = {'POS': 'ADJ'}
COMPOUND_ADJ = {**ADJ, 'POS': '+'}
VERB = {'POS': 'VERB'}
COMPOUND_VERB = {**VERB, 'POS': '+'}


def noun(compound: bool = False, adj: bool = False):
    if compound and adj:
        return {**COMPOUND_ADJ, **COMPOUND_NOUN}
    elif compound:
        return COMPOUND_NOUN
    elif adj:
        return {**COMPOUND_ADJ, **NOUN}
    else:
        return NOUN


# Pymusas Tags: https://ucrel.github.io/pymusas/ , https://ucrel.lancs.ac.uk/usas/usas_guide.pdf
def pymusas_tag() -> dict:
    """ Generate a spacy matcher pattern for pymusas tags based on arguments. """
    return dict()  # todo: design the arguments, e.g. > 1 tag, top level tag - maybe use enums.
