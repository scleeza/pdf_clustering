import streamlit as st
import spacy_streamlit
import en_core_web_sm
from transformers import pipeline, AutoTokenizer


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
    spacy_streamlit.visualize_ner(doc,
                                  labels=nlp.get_pipe("ner").labels,
                                  title="spaCy NER",
                                  sidebar_title=None,
                                  show_table=False)