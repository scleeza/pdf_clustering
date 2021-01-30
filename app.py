import streamlit as st
from pathlib import Path
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from page_dataload import load_data
from page_show_data import build_wordcloud
from page_text_clean import clean_data
from page_LDA import run_LDA, cluster_data, fit_best_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def main():
    state = _get_state()
    st.sidebar.info('streamlit version: {}'.format(st.__version__))
    st.sidebar.title("What to do")
    if st.sidebar.button("Initial Settings",key="initial_bt"):
        initial_state(state)

    pages = {"Main": main_app,
             "Setting": setting
             }
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))
    pages[page](state)
    state.sync()


def main_app(state):
    # header
    header = read_markdown_file("markdowns/header.md")
    st.markdown(header, unsafe_allow_html=True)
    # procedure
    _expander = st.beta_expander("PROCEDURE",expanded=True)
    _expander.markdown(read_markdown_file("markdowns/dataload.md"),
                       unsafe_allow_html=True)
    # LDA example
    lda_expander = st.beta_expander("How LDA do")
    lda_expander.markdown((read_markdown_file("markdowns/LDA.md")),
                          unsafe_allow_html=True)
    st.title("Input Data:")
    load_df(state)
    st.title("Cleaned Data:")
    clean_df(state)
    st.title('Clustered Data:')
    cluster_df(state)
    st.title('Fit Data:')
    fit_df(state)

def load_df(state):
    if state.df is None:
        st.info('There is no files uploaded, Please input data, select a way to do so')
        state.upload_way = st.radio("Chose one way to upload file:",("By dataframe","By PDFs"))
        load_data(state)
    else:
        st.dataframe(state.df)
        #st.plotly_chart(plotly_table(state.df))
        if st.button("Reset Data",key="reset_bt"):
            state.df =None
            state.df_clean =None



def clean_df(state):
    if state.df is None:
        st.info("There is no files uploaded, Please input data at Input Data block")
    else:
        state.column = st.selectbox("Chose Columnn Names:", list(state.df.columns),key="clean_bt")
        if st.button('Clean'):
            clean_data(state)
    if state.df_clean is not None:
        st.write('Cleaned Data:')
        st.dataframe(state.df_clean.head(3))

def cluster_df(state):
    if state.df_clean is None:
        st.info("There is no files uploaded, Please input data at Input Data block")
    else:
        st.write("Number of topics:{}".format(state.lda_topics))
        if st.button('Run Trials', key='cluster_bt'):
            with st.spinner("Clustering"):
                state.scores,state.lda_models, state.corpus = cluster_data(state)
                st.success("Done!")
        try:
            TOPICS_LIST = range(1, state.lda_topics + 1)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.lineplot(x=TOPICS_LIST, y=state.scores, ax=ax)
            plt.title('Coherence_scores')
            plt.xlabel('TOPICS')
            plt.ylabel('Scores')
            st.pyplot(fig)
        except:
            st.info("Press run trials")


def fit_df(state):
    if state.df_clean is None:
        st.info("There is no files uploaded, Please input data at Input Data block")
    else:
        chose_topic = st.number_input("Chose best topics", min_value=1, max_value=10, value=5)
        state.chice_topic = chose_topic
        if st.button("Fit LDA models", key='lda_fit'):
            topic_lst = fit_best_model(chose_topic, lda_models=state.lda_models, corpus=state.corpus)
            state.df_clean['topic'] = topic_lst
        wordcloud = build_wordcloud(state.df_clean.loc[state.df_clean.topic==2,:], state.column)
        fig = plt.figure(figsize=(6, 6), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)

def show_wordcloud(state):
    if state.df_clean is None:
        st.info("There is no files uploaded, Please input data at Input Data block")
    else:
        wordcloud = build_wordcloud(state.df_clean.loc[state.df_clean.topic == 2, :], state.column)
        fig = plt.figure(figsize=(6, 6), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)



def initial_state(state):
    if state.intial is None:
        state.pages_read = 10
        state.pos_tag = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']
        state.lda_topics = 5
        state.min_count = 5
        state.threshold = 10
        state.initial = True


def setting_test(state):
    initial_state(state)
    st.sidebar.subheader("Import Setting")
    pages_read = st.sidebar.text_input('Read how many pages per file (avoid running too long)',
                                     value=str(state.pages_read))
    state.pages_read = int(pages_read)
    st.sidebar.subheader("Clean Setting")
    state.pos_tag = st.sidebar.multiselect(label='Choose POS Tag:',
                                           options=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM','NUM'],
                                           default=state.pos_tag)
    st.sidebar.subheader("LDA Topics")
    state.lda_topics = st.sidebar.number_input("# of Topics you want to try",min_value=1,max_value=10,value=state.lda_topics)

    st.sidebar.subheader("Bigram setting")
    min_count = st.sidebar.text_input('Min Occupancy',
                               value=str(state.min_count))
    state.min_count = int(min_count)
    threshold = st.sidebar.text_input('Threshold',
                               value=str(state.threshold))
    state.threshold = threshold

def setting(state):
    st.title("Setting")
    st.subheader("Import Setting")
    pages_read = st.text_input('Read how many pages per file (avoid running too long)',
                                     value=str(state.pages_read))

    st.subheader("Clean Setting")
    pos_tag = st.multiselect(label='Choose POS Tag:',
                                           options=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM','NUM'],
                                           default=state.pos_tag)
    st.subheader("LDA Topics")
    lda_topics = st.number_input("# of Topics you want to try",min_value=1,max_value=10,value=state.lda_topics)

    st.subheader("Bigram setting")
    min_count = st.text_input('Min Occupancy',
                               value=str(state.min_count))

    threshold = st.text_input('Threshold',
                               value=str(state.threshold))


    state.pages_read = int(pages_read)
    state.post_tag = pos_tag
    state.lda_topics =lda_topics
    state.min_count = int(min_count)
    state.threshold = threshold

# Helper functions

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


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


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def run_the_instruction(state):
    st.write('Paste text into context column and type question you want to know')


@st.cache
def get_dataframe(state):
    return state.df


def run_the_exp(state):
    placeholder = st.empty()
    options = st.sidebar.radio('Choose expression', ('raw_text', 'NER'))
    with placeholder.beta_container():
        if options =='raw_text':
            st.write("This is one element")
            st.write(state.df)
        elif options =='NER':
            st.write("This is another")


if __name__ == "__main__":
    main()