import streamlit as st
import spacy_streamlit
import spacy
from transformers import pipeline, AutoTokenizer


def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "NLP", "Experiment"])
    if app_mode == "Show instructions":
        #st.sidebar.success('To continue select "Run the app".')
        st.write('Paste text into context column and type question you want to know')
    elif app_mode == "Experiment":
        run_the_exp()
    elif app_mode == "NLP":
        run_the_app()


@st.cache(allow_output_mutation=True)
def load_Roberta_model():
    model_name = "deepset/roberta-base-squad2"
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return qa_model

def get_answer(context,question,model):
    QA_input = {
        'question': question,
        'context': context
    }
    res = model(QA_input)
    return res['answer']


def run_the_exp():
    placeholder = st.empty()
    options = st.sidebar.radio('Choose expression',('raw_text', 'NER'))
    with placeholder.beta_container():
        if options =='raw_text':
            st.write("This is one element")
        elif options =='NER':
            st.write("This is another")


@st.cache(allow_output_mutation=True)
def get_questions():
    return []


def run_the_app():
    st.title('Feature Extraction based on Roberta model')
    with st.spinner('Wait for it...'):
        qa_model = load_Roberta_model()
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
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(context)
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels,title="Context_NER",
        sidebar_title=None,show_table=False)





if __name__ == "__main__":
    main()