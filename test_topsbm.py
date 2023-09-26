import time

import torch
import graph_tool

SLEEP = 1e6

if __name__ == '__main__':
    import pandas as pd
    # print(f"Sleeping for: {SLEEP}s")
    # time.sleep(SLEEP)
    from juxtorpus.corpus import Corpus, Corpora
    from juxtorpus.corpus.meta import SeriesMeta
    from juxtorpus.tm import TopSBM
    import graph_tool

    # from transformers import pipeline
    #
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    # print(unmasker("Hello I'm a [MASK] model."))
    print("OpenMP enabled: ", graph_tool.openmp_enabled())

    titles = pd.read_table("./tests/assets/topsbm/titles.txt", header=None)
    corpus = Corpus.from_dataframe(pd.read_table("./tests/assets/topsbm/corpus.txt", header=None), col_doc=0)
    corpus.add_meta(SeriesMeta(id_='title', series=titles.loc[:, 0]))

    print("+ TopSBM")
    tsbm = TopSBM(corpus, meta_title='title')
    tsbm.build()
    print("+ Successful.")
