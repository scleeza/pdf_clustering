import streamlit as st
import spacy_streamlit
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import en_core_web_sm
from transformers import pipeline, AutoTokenizer
import pandas as pd
from wikiscraper import wiki_scraper


def main():
    # main gate of app, showing menu on main page
    state = _get_state()
    st.sidebar.title("What to do")
    pages = {
        "Show instructions": run_the_instruction,
        "Load Data": run_the_dataloader,
        "NLP": run_the_nlp,
        "Experiment": run_the_exp
    }
    page = st.sidebar.selectbox("Choose the app mode", list(pages.keys()))
    pages[page](state)
    state.sync()


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


@st.cache(allow_output_mutation=True)
def get_questions():
    return []


# loading bert model, using cache to save time
@st.cache(allow_output_mutation=True)
def load_roberta_model():
    model_name = "deepset/roberta-base-squad2"
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return qa_model


# Q+A model
def get_answer(context, question, model):
    QA_input = {
        'question': question,
        'context': context
    }
    res = model(QA_input)
    return res['answer']


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


def run_the_nlp(state):
    st.title('Feature Extraction based on Roberta model')
    with st.spinner('Wait for it...'):
        qa_model = load_roberta_model()
    # list of questions
    context_region = st.empty()

    context = context_region.text_area(label='Context')
    question = st.text_area(label='Question')
    if st.button('Query'):
        if question not in get_questions():
            get_questions().append(question)
        answer = get_answer(context=context, question=question, model=qa_model)
        with st.beta_expander(label=question):
            st.write(answer)
    # for i in get_questions():
    #     st.write(i)

    # left_column, right_column = st.beta_columns(2)
    # context = left_column.text_area(label='''
    # Context of Qustions
    # ''')
    # question = right_column.text_area(label='''
    # Qustions
    # ''')
    # if left_column.button('Query'):
    #     answer = get_answer(context=context, question=question, model=qa_model)
    #     st.write(answer)
    nlp = en_core_web_sm.load()
    doc = nlp(context)
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="spaCy NER", sidebar_title=None, show_table=False)


def run_the_dataloader(state):
    state.url = st.text_input('URL')
    if state.url != "":
        state.df = wiki_scraper(state.url)
        st.write(state.df)


if __name__ == "__main__":
    main()