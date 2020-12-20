from wikiscraper import wiki_scraper
import streamlit as st


def run_the_dataloader(state):
    state.url = st.text_input('URL')
    if state.url != "":
        state.df = wiki_scraper(state.url)
        st.write(state.df)