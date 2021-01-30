"""Functions related to load data"""
import streamlit as st
import os
import PyPDF2
import pandas as pd

# Define parameters
FILE_TYPES = ['csv', 'pkl']

# main function of loading data
def load_data(state):
    if state.upload_way == "By dataframe":
        file = st.file_uploader("Select file", type=FILE_TYPES)
        if not file:
            st.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

        else:
            if file.name.endswith('csv'):
                df = pd.read_csv(file)
                if st.button('Upload', key='file_upload_csv'):
                    state.df = df

            elif file.name.endswith('pkl'):
                df = pd.read_pickle(file)
                if st.button('Upload', key='file_upload_pkl'):
                    state.df = df

            file.close()

    else:
        folder_path = st.text_input('Path of directory')
        upload_but = st.button('Upload', key='folder_upload')
        if folder_path != '' and upload_but:
            with st.spinner("PDF text extracting..."):
                try:
                    df_extracted = read_all_pdfs(folder_path, int(state.pages_read))
                    state.df = pdf2df(df_extracted)

                except:
                    st.info("Input valid folder path, please try again")

# Helper functions
def text_input(label):
    return st.text_input(label)



def read_all_pdfs(FILE_PATH, pages_read):
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
                for page in range(pages_read):
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
    return extraction_pdfs


def pdf2df(extraction_pdfs):
    # transfer unreadable_pages list into a string list in order to create dataframes
    # for k, v in extraction_pdfs.items():
    #     if v['encrpyted'] == False:
    #         str_list = [str(page) for page in v['unextractable_pages']]
    #         v['unextractable_pages'] = combine_texts(str_list)
    extraction_df = pd.DataFrame.from_dict(
        {k: [v['pages'], v['unextractable_pages'], v['docs']] for k, v in extraction_pdfs.items() if
         v['encrpyted'] == False}).transpose()
    extraction_df.columns = ['pages', 'unextractable_pages', 'docs']
    #extraction_df[extraction_df['doc'].apply(len)>40]

    return extraction_df



def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text
