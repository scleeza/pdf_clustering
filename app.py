import streamlit as st
from wikiscraper import wiki_scraper
import base64

url = st.text_input('please input url!')
if len(url) > 0:
    df = wiki_scraper(url)
    st.dataframe(df)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="wiki_table.csv">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

