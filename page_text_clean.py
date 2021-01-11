import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from page_show_data import plot_cloud
import en_core_web_sm


def run_text_clean(state):
    st.title('Text cleaning')
    allowed_pos_tag = st.sidebar.multiselect(label='2..Choose POS Tag:',
                                            options = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'],
                                            default = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'])

    nlp = en_core_web_sm.load()
    clean_text = lambda x: clean_text_pipe(x,allowed_postags=allowed_pos_tag,nlp=nlp)
    if st.sidebar.button('Clean'):
        with st.spinner('Wait for it...'):
            df_clean = pd.DataFrame(state.df[state.text_col_name].apply(clean_text))
        st.write(df_clean)
        text = combine_texts(df_clean[state.text_col_name].tolist())
        wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black', colormap='Set2',
                              collocations=False, stopwords=STOPWORDS).generate(text)
        fig = plot_cloud(wordcloud)
        st.header('WordCloud')
        st.pyplot(fig)
        state.df_clean = df_clean
    # if context != '':
    #     doc = nlp(context)
    #     spacy_streamlit.visualize_ner(doc,
    #                                   labels=nlp.get_pipe("ner").labels,
    #                                   title="spaCy NER",
    #                                   sidebar_title=None,
    #                                   show_table=False)
    #
    #     clean_func = lambda x: clean_text_pipe(x, nlp, allowed_postags=allowed_postag)
    #     st.write(clean_func(context))


def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text


def chunck_list(lst, chunck_size=5000):
    '''Splitting large doc into batches, defualt batch size 5000.'''
    for i in range(0, len(lst), chunck_size):
        yield lst[i:i + chunck_size]


def clean_text_pipe(text, allowed_postags, nlp):
    '''Remove stop words and punctuaion'''
    BATCH_SIZE = 5000
    if len(text) > BATCH_SIZE:
        split_text = chunck_list(text)
        docs = [nlp(t) for t in split_text]
        cleaned_docs = []
        for doc in docs:
            cleaned_doc = [token.lemma_ for token in doc if
                           not token.is_stop and not token.is_punct and token.pos_ in allowed_postags]
            cleaned_text = ' '.join(cleaned_doc)
            cleaned_docs.append(cleaned_text)
        return combine_texts(cleaned_docs)
    else:
        doc = nlp(text)
        cleaned_doc = [token.lemma_ for token in doc if
                       not token.is_stop and not token.is_punct and token.pos_ in allowed_postags]
        cleaned_text = ' '.join(cleaned_doc)
        return cleaned_text


