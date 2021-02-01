import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from page_show_data import plot_cloud
import en_core_web_sm


def clean_data(state):
    nlp = en_core_web_sm.load()
    clean_text = lambda x: clean_text_pipe(x, allowed_postags=state.pos_tag, nlp=nlp)
    with st.spinner('Wait for it...'):
        try:
            state.df[state.column] = state.df[state.column].map(clean_text)
            st.success("data cleaned!")
        except:
            st.info('Something went wrong, check log')



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
    BATCH_SIZE = 1000000
    if len(text) > BATCH_SIZE:
        split_text = chunck_list(text)
        docs = [nlp(t) for t in split_text]
        cleaned_docs = []
        for doc in docs:
            cleaned_doc = [token.lemma_.lower() for token in doc if
                           not token.is_stop and not token.is_punct and token.pos_ in allowed_postags and token.is_alpha]
            cleaned_text = ' '.join(cleaned_doc)
            cleaned_docs.append(cleaned_text)
        return combine_texts(cleaned_docs)
    else:
        doc = nlp(text)
        cleaned_doc = [token.lemma_.lower() for token in doc if
                       not token.is_stop and not token.is_punct and token.pos_ in allowed_postags and token.is_alpha]
        cleaned_text = ' '.join(cleaned_doc)
        return cleaned_text


