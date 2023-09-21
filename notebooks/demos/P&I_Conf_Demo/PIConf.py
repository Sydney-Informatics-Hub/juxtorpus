import re
import difflib
import numpy as np

from collections import Counter
from wordcloud import WordCloud
import panel as pn
import panel.widgets as pnw
import holoviews as hv
from holoviews import dim
from bokeh.palettes import Category20c
from bokeh.models import HoverTool

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.corpora import Corpora
from juxtorpus.corpus.meta import SeriesMeta
import warnings
warnings.filterwarnings('ignore')

hv.extension('bokeh')


def diff_corpus(corpus, group_col, order_col):
    df = corpus.to_dataframe()
    rm_df = df.copy()
    add_df = df.copy()
    for gn, g in df.groupby(group_col):
        g = g.sort_values(order_col, ascending=True)
        for idx, s in g[order_col].items():
            if s == g[order_col].min():
                corpus._df.loc[idx, 'add'] = ''
                corpus._df.loc[idx, 'rm'] = ''
                rm_df.loc[idx, 'document'] = ''
                add_df.loc[idx, 'document'] = ''
                prev_text = g.loc[idx, 'document']
                continue
            curr_text = g.loc[idx, 'document']
            rm_df.loc[idx, 'document'], add_df.loc[idx, 'document'] = diff_texts(prev_text, curr_text)
            corpus._df.loc[idx, 'add'] = add_df.loc[idx, 'document']
            corpus._df.loc[idx, 'rm'] = rm_df.loc[idx, 'document']
            prev_text = curr_text

    rm_corpus = Corpus.from_dataframe(rm_df, col_doc='document', name=corpus.name + '_rm')
    add_corpus = Corpus.from_dataframe(add_df, col_doc='document', name=corpus.name + '_add')

    return rm_corpus, add_corpus, corpus


def diff_texts(t1, t2):
    whitespace = re.compile(r'\s+')
    t1 = whitespace.split(t1)
    t2 = whitespace.split(t2)
    seqmatcher = difflib.SequenceMatcher(isjunk=None, a=t1, b=t2, autojunk=False)
    out_rm, out_add = [], []
    for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'delete':
            out_rm += t1[a0:a1 + 1]
        if tag == 'insert':
            out_add += t2[b0:b1 + 1]
        if tag == 'replace':
            out_rm += t1[a0:a1 + 1]
            out_add += t2[b0:b1 + 1]
    return ' '.join(out_rm), ' '.join(out_add)


def doc_compare(corpus, prev_idx, curr_idx):
    vocab = corpus.dtm.term_names
    dtm = corpus.dtm.matrix[[curr_idx, prev_idx], :].astype(bool)
    new_idx = (dtm[0, :] > dtm[1, :]).indices
    rm_idx = (dtm[0, :] < dtm[1, :]).indices
    new_dict = dict(zip(vocab[new_idx], corpus.dtm.matrix[curr_idx, new_idx].toarray().flatten()))
    rm_dict = dict(zip(vocab[rm_idx], corpus.dtm.matrix[prev_idx, rm_idx].toarray().flatten()))
    return new_dict, rm_dict


def upload_data(corpora):
    return corpora.widget()


def process_corpus(corpora):
    # This example assumes we have only uploaded one corpus
    corpus_name = corpora.items()[0]
    corpus = corpora[corpus_name]
    corpus_rm, corpus_add, updated_corpus = diff_corpus(corpus, 'site_hostname', 'version')
    corpora.add([corpus_rm, corpus_add])

    s = corpus.meta.get('version').series.copy()
    for i, _ in s.items():
        s[i] = {}

    new_meta = SeriesMeta(id_='newwords', series=s.copy())
    rm_meta = SeriesMeta(id_='rmwords', series=s.copy())

    sites = corpus.meta.get('site_hostname').series.values

    for s in sites:
        site_corpus = corpus.slicer.filter_by_item('site_hostname', s)
        vers = site_corpus.meta.get('version').series.sort_values(ascending=True)
        prev_idx = 0
        for idx, ver in vers.items():
            if ver == 1:
                prev_idx = idx
                continue
            new_meta.series[idx], rm_meta.series[idx] = doc_compare(corpus, idx, prev_idx)
            prev_idx = idx
    corpus.add_meta(new_meta)
    corpus.add_meta(rm_meta)

    return corpus


def visualise(corpora):
    corpus = process_corpus(corpora)
    # Load the data as a pandas dataframe
    df = corpus.to_dataframe()
    # Compute word sizes for newwords and rmwords
    df['add_token'] = df['add'].apply(lambda t: len(t.split(' ')))
    df['rm_token'] = df['rm'].apply(lambda t: -len(t.split(' ')))
    df['short_add'] = df['add'].str[:100] + np.where(df['add'].str.len() > 100, '...', '')
    df['short_rm'] = df['rm'].str[:100] + np.where(df['rm'].str.len() > 100, '...', '')

    # Dropdown for category selection
    category_dropdown = pnw.Select(name='Category', options=df['Category'].unique().tolist(),
                                   value=df['Category'].sample(1).values[0])

    # Define unique_sites and color_map
    unique_sites = sorted(df['site_hostname'].unique())
    color_map = {site: Category20c[20][i % 20] for i, site in enumerate(unique_sites)}

    # MultiSelect dropdown for site_hostname selection
    site_hostname_dropdown = pnw.MultiSelect(name='Websites', options=list(unique_sites), value=list(unique_sites),
                                             size=4)

    # Extract unique years from df
    unique_years = sorted(df['year'].unique())

    # Create a dropdown for year selection
    year_dropdown = pnw.MultiSelect(name='Year', options=unique_years, value=unique_years,
                                    size=4)  # default to all years

    # Function to update site_hostname dropdown options based on selected category
    def update_site_dropdown(category):
        relevant_sites = df[df['Category'] == category]['site_hostname'].unique().tolist()
        site_hostname_dropdown.options = sorted(relevant_sites)
        site_hostname_dropdown.value = relevant_sites

    # Set initial values for site_hostname dropdown
    update_site_dropdown(category_dropdown.value)

    # Watch changes to category_dropdown and update site_hostname dropdown accordingly
    category_dropdown.param.watch(lambda event: update_site_dropdown(event.new), 'value')

    # Function to update year dropdown options based on selected category and sites
    def update_year_dropdown(category, sites):
        relevant_years = df[(df['Category'] == category) & (df['site_hostname'].isin(sites))]['year'].unique().tolist()
        year_dropdown.options = sorted(relevant_years)
        year_dropdown.value = relevant_years

    # Watch changes to site_hostname_dropdown and update year_dropdown accordingly
    site_hostname_dropdown.param.watch(lambda event: update_year_dropdown(category_dropdown.value, event.new), 'value')

    # Set the initial values for year_dropdown
    update_year_dropdown(category_dropdown.value, site_hostname_dropdown.value)

    # Refactor the function to apply filtering on the original DataFrame directly and customize the tooltip
    @pn.depends(category_dropdown.param.value, site_hostname_dropdown.param.value, year_dropdown.param.value)
    def update_plot_with_custom_tooltip_refactored(selected_category, selected_sites, selected_year):
        # Filter data based on selected category and site_hostnames from the original DataFrame

        filtered_df = df[(df['Category'] == selected_category) \
                         & (df['site_hostname'].isin(selected_sites)) \
                         & (df['year'].isin(selected_year))]

        # Define tooltips
        tooltips_newwords = [
            ("Contents", "@short_add"),
            ("Time", "@timestamp{%Y-%m-%d}"),  # Adjusted format for timestamp
            ("Site", "@site_hostname")

        ]

        tooltips_rmwords = [
            ("Contents", "@short_rm"),
            ("Time", "@timestamp{%Y-%m-%d}"),  # Adjusted format for timestamp
            ("Site", "@site_hostname")

        ]

        # Bars for newwords with color mapping and custom tooltip
        bars_newwords = hv.Bars(
            filtered_df, kdims=['year', 'site_hostname'], vdims=['add_token', 'short_add', 'timestamp']
        ).opts(
            color=dim('site_hostname').categorize(color_map),
            tools=[HoverTool(tooltips=tooltips_newwords, formatters={'@timestamp': 'datetime'})]
            # Explicitly format timestamp as datetime
        )

        bars_rmwords = hv.Bars(
            filtered_df, kdims=['year', 'site_hostname'], vdims=['rm_token', 'short_rm', 'timestamp']
        ).opts(
            color=dim('site_hostname').categorize(color_map),
            tools=[HoverTool(tooltips=tooltips_rmwords, formatters={'@timestamp': 'datetime'})]
            # Explicitly format timestamp as datetime
        )

        # Overlay the two bar plots
        overlay_bars = (bars_newwords * bars_rmwords).opts(
            width=800, height=400, xlabel='Year', ylabel='Word Count Changes', show_legend=True
        ).opts(xlabel='Year')

        return overlay_bars

    # Function to generate a word cloud from a dictionary and convert it to an RGB image
    def generate_wordcloud_image(data_dict):
        if len(data_dict) > 1:
            wc = WordCloud(background_color='white').generate_from_frequencies(data_dict)
        else:
            wc = WordCloud(background_color='white').generate_from_frequencies({' ': 1})
        return np.array(wc.to_image())

    # Function to generate a combined word cloud from a DataFrame
    def generate_combined_wordcloud_image(df, column_name):
        combined_dict = Counter({})
        for index, row in df.iterrows():
            combined_dict.update(row[column_name])
        return generate_wordcloud_image(combined_dict), len(combined_dict.keys())

    # Function to generate word clouds based on the current filtered DataFrame
    @pn.depends(category_dropdown.param.value, site_hostname_dropdown.param.value, year_dropdown.param.value)
    def display_wordclouds(selected_category, selected_sites, selected_years):
        filtered_df = df[(df['Category'] == selected_category) & (df['site_hostname'].isin(selected_sites)) & (
            df['year'].isin(selected_years))]

        # Generate word cloud images for newwords and rmwords
        newwords_image, newwords_no = generate_combined_wordcloud_image(filtered_df, 'newwords')
        rmwords_image, rmwords_no = generate_combined_wordcloud_image(filtered_df, 'rmwords')

        # Create HoloViews elements for the word clouds
        newwords_cloud = hv.RGB(newwords_image).opts(title=f'New Words: {newwords_no}', width=500, height=300)
        rmwords_cloud = hv.RGB(rmwords_image).opts(title=f'Removed Words: {rmwords_no}', width=500, height=300)

        return (newwords_cloud + rmwords_cloud).cols(1)

    # Combine everything into a dashboard
    layout = pn.Row(
        pn.Column(pn.Row(category_dropdown, site_hostname_dropdown, year_dropdown),
                  update_plot_with_custom_tooltip_refactored),
        display_wordclouds
    )

    return layout.servable()
