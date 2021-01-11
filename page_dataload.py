
import streamlit as st
from pathlib import Path
import os
import PyPDF2
import pandas as pd

FILE_TYPES =['csv', 'pkl']

def run_the_dataloader(state):
    ''' run the function loading data '''
    intro_markdown = read_markdown_file("dataload.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
    st.title("Import Files:")
    # declare expander component
    upload_expander = st.beta_expander('Upload by file')
    file = upload_expander.file_uploader("Select file", type=FILE_TYPES)
    folder_expander = st.beta_expander('Upload by folder')
    show_file = upload_expander.empty()

    with upload_expander:
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        elif file.name.endswith('csv'):
            df = pd.read_csv(file)
            if st.button('Upload', key='file_upload_csv'):
                state.df = df
                st.write(state.df)
        elif file.name.endswith('pkl'):
            df = pd.read_pickle(file)
            if st.button('Upload', key='file_upload_pkl'):
                state.df = df
                st.write(state.df)
    with folder_expander:
        folder_path = st.text_input('Path of directory')
        page_read = st.text_input('Read how many pages per file (avoid running too long)', value='10')
        upload_but = st.button('Upload', key='folder_upload')
        if folder_path != '' and upload_but:
            read_all_pdfs(folder_path, int(page_read))
            st.write(state.df)
    if file:
        file.close()






def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def text_input(label):
    return st.text_input(label)


@st.cache
def get_dataframe(state):
    return state.df

def read_all_pdfs(FILE_PATH,page_read):
    '''read all pdf files from given folder path and return '''
    file_list = [f for f in os.listdir(path=FILE_PATH) if f.endswith('.pdf') or f.endswith('.PDF')]
    # PDF extraction
    # imformations we want to extract
    extraction_pdfs = {}

    for file_name in file_list:
        # save the results
        unreadable_pages = []
        list_of_text = []
        buff_dict = {}
        # file path
        path = FILE_PATH + '/' + file_name
        # create pdf reader object
        with open(path, 'rb') as pdfFileObj:

            # file reader
            try:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False)
                # number of pages
                pages = pdfReader.numPages
                buff_dict['encrpyted'] = False
                buff_dict['pages'] = pages
                # loop over all pages and extract text, if not extractable, the page number will be recored
                for page in range(page_read):
                    pageObj = pdfReader.getPage(page)
                    try:
                        text_per_page = pageObj.extractText()
                        list_of_text.append(text_per_page)
                    except:
                        unreadable_pages.append(page)
                        continue
                # adding unextractable page numbers into dictionary
                buff_dict['unextractable_pages'] = unreadable_pages

                # combine texts of each pages into a whole texts or called doc
                buff_dict['docs'] = combine_texts(list_of_text)
            except:
                buff_dict['encrpyted'] = True
                extraction_pdfs[file_name] = buff_dict
                continue

        extraction_pdfs[file_name] = buff_dict

# some helper function
def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text
