import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import en_core_web_sm




def run_show_data(state):
    """ run data exploratory page"""
    st.title('Data Pre-processing')
    st.write('Choose 1.Column and 2.POS tags and 3.press clean')
    dataframe_expander = st.beta_expander('Loaded Data')
    state.text_col_name = st.sidebar.selectbox("1.Choose text column:", list(state.df.columns))
    with dataframe_expander:
        st.write('Dataframe')
        st.dataframe(state.df)
        st.write('Column info')
        st.write(state.df[state.text_col_name].apply(len).describe())
    run_text_clean(state)


def plot_cloud(wordcloud):
    # Set figure size
    fig = plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")
    return fig


def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text


def run_text_clean(state):
    cleaned_dataframe = st.beta_expander('Cleaned Data')
    allowed_pos_tag = st.sidebar.multiselect(label='2..Choose POS Tag:',
                                            options = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'],
                                            default = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'])
    cleaned_dataspace = cleaned_dataframe.empty()
    nlp = en_core_web_sm.load()
    clean_text = lambda x: clean_text_pipe(x,allowed_postags=allowed_pos_tag,nlp=nlp)
    if st.sidebar.button('Clean'):
        with cleaned_dataframe:
            with st.spinner('Wait for it...'):
                df_clean = pd.DataFrame(state.df[state.text_col_name].apply(clean_text))
                state.df_clean = df_clean
                st.write(df_clean)
    else:
        cleaned_dataspace.info('Choose 1.Column and 2.POS tags and 3.press clean')


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


def plotly_table(df):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig


