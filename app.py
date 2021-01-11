import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from page_dataload import run_the_dataloader
from page_show_data import run_show_data
from page_text_clean import run_text_clean
from page_LDA import run_LDA



def main():
    st.sidebar.info('streamlit version: {}'.format(st.__version__))
    st.sidebar.title("What to do")
    state = _get_state()

    pages = {
        "1.Load Data": run_the_dataloader,
        "2.Data Prepare": run_show_data,
        "3.LDA": run_LDA
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