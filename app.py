import streamlit as st
from pathlib import Path
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from page_dataload import load_data
from page_show_data import build_wordcloud, plot_cloud
from page_text_clean import clean_data
from page_LDA import run_LDA, cluster_data, fit_best_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def main():
    state = _get_state()
    st.sidebar.info('streamlit version: {}'.format(st.__version__))
    st.sidebar.title("What to do")

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
    _expander = st.beta_expander("PROCEDURE", expanded=False)
    _expander.markdown(read_markdown_file("markdowns/dataload.md"),
                       unsafe_allow_html=True)
    # LDA example
    lda_expander = st.beta_expander("How LDA do")
    lda_expander.markdown((read_markdown_file("markdowns/LDA.md")),
                          unsafe_allow_html=True)
    st.markdown("---")
    funcs = {"1.Load Data": load_df,
             "2.Clean Data": clean_df,
             "3.Cluster Data": cluster_df,
             "4.Draw Word Cloud": show_wordcloud
             }
    func = st.sidebar.selectbox("Select functions", tuple(funcs.keys()))
    funcs[func](state)

    if st.sidebar.button("Initial Settings", key="initial_bt"):
        initial_state(state)
    st.markdown("---")
    st.title("Data:")
    st.write(state.df)
    if st.button("Reset Data", key="reset_bt"):
        state.df = None
        state.df_clean = None
    st.markdown("---")
    st.title("World Cloud:")
    all_cloud, one_cloud = st.beta_columns(2)
    all_cloud.write(state.fig)
    one_cloud.write(state.one)
    st.markdown("---")


def load_df(state):
    st.title("Load Data:")
    if state.df is None:
        state.upload_way = st.radio("Chose one way to upload file:",("By dataframe","By PDFs"))
        load_data(state)
    else:
        st.info("Data has been load")



def clean_df(state):
    st.title("Clean Data:")
    if state.df is None:
        st.info("There is no files uploaded, Please load data first")
    else:
        state.column = st.selectbox("Chose Columnn Names:", list(state.df.columns),key="clean_bt")
        if st.button('Clean'):
            clean_data(state)
    # if state.df_clean is not None:
    #     st.write('Cleaned Data:')
    #     st.dataframe(state.df_clean.head(3))

def cluster_df(state):
    st.title("Cluster Data:")
    if state.df is None:
        st.info("There is no files uploaded, Please load data first")
    else:
        c1,c2 = st.beta_columns((2,1))
        c1.write("run {} clusters test" .format(state.lda_topics))
        if c1.button('Run Trials', key='cluster_bt'):
            with st.spinner("Clustering"):
                state.scores,state.lda_models, state.corpus = cluster_data(state)
                c1.success("Done!")
        try:
            TOPICS_LIST = range(1, state.lda_topics + 1)
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.lineplot(x=TOPICS_LIST, y=state.scores, ax=ax)
            plt.title('Coherence_scores')
            plt.xlabel('TOPICS')
            plt.ylabel('Scores')
            c1.pyplot(fig)
        except:
            c1.info("Press run trials")
        state.chose_num = c1.slider("Chose best cluster number, the one with largest score",min_value=1,max_value=10,value=state.chose_num)
        if c1.button("Fit LDA models", key='lda_fit'):
            topic_lst = fit_best_model(state.chose_num, lda_models=state.lda_models, corpus=state.corpus)
            state.df['topic'] = topic_lst
            c1.success("Done")

# def fit_df(state):
#     if state.df_clean is None:
#         st.info("There is no files uploaded, Please load data first")
#     else:
#         chose_topic = st.number_input("Chose best topics", min_value=1, max_value=10, value=5)
#         state.chose_topic = chose_topic
#         if st.button("Fit LDA models", key='lda_fit'):
#             topic_lst = fit_best_model(chose_topic, lda_models=state.lda_models, corpus=state.corpus)
#             state.df_clean['topic'] = topic_lst


def show_wordcloud(state):
    st.subheader('Word cloud')
    if state.df is None:
        st.info("There is no files uploaded, Please load data first")
    else:
        # wordcloud = build_wordcloud(state.df, state.column)
        # state.fig = plot_cloud(wordcloud)

        specific_one = st.selectbox("select topic index",list(range(state.chose_num)))
        if st.button('Plot',key="word_plot"):
            one_cloud = build_wordcloud(state.df.loc[state.df['topic'] == int(specific_one), :], state.column)
            state.one = plot_cloud(one_cloud)


def initial_state(state):
    if state.intial is None:
        state.pages_read = 10
        state.pos_tag = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']
        state.lda_topics = 5
        state.chose_num = 5
        state.min_count = 5
        state.threshold = 10
        state.initial = True
        state.no_below = 5
        state.no_above = 500


def setting(state):
    st.title("Setting")
    st.subheader("Import Setting")
    pages_read = st.text_input('Read how many pages per file (avoid running too long)',
                                     value=str(state.pages_read))

    st.subheader("Clean Setting")
    pos_tag = st.multiselect(label='Choose POS Tag:',
                                           options=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM','NUM'],
                                           default=state.pos_tag)
    st.subheader("Embedding Model:")

    st.subheader("LDA Topics")
    lda_topics = st.slider("# of Topics you want to try",min_value=1,max_value=10,value=state.lda_topics)

    st. subheader('Word Frequency filter:')
    no_below = st.slider("Select Word that at least appear N times",min_value=1,max_value=100,value=state.no_below)
    no_above = st.slider("select Word that not appear over N times",min_value=200,max_value=1000,value=state.no_above)
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
    state.no_below = no_below
    state.no_above = no_above

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