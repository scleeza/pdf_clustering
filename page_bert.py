import streamlit as st
import spacy_streamlit
import en_core_web_sm
from transformers import pipeline
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer


@st.cache(allow_output_mutation=True)
def get_questions():
    return []


# loading bert model, using cache to save time
@st.cache(allow_output_mutation=True)
def load_roberta_model():
    model_name = "deepset/roberta-base-squad2"
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return qa_model
    # configuration = RobertaConfig()
    # model = RobertaModel(configuration)
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # return model, tokenizer


# Q+A model
def get_answer(context, question, model):
    QA_input = {
        'question': question,
        'context': context
    }
    res = model(QA_input)
    return res['answer']


def run_the_nlp(state):
    st.title('spaCy cleaning')
    context = st.text_area('Paste some text to test:')
    allowed_postag = st.sidebar.multiselect(label='Pos_tag',
                                            options=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'],
                                            default=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'AUX', 'ADP', 'SYM', 'NUM'],)
    nlp = en_core_web_sm.load()
    if context !='':
        doc = nlp(context)
        spacy_streamlit.visualize_ner(doc,
                                      labels=nlp.get_pipe("ner").labels,
                                      title="spaCy NER",
                                      sidebar_title=None,
                                      show_table=False)

        clean_func = lambda x: clean_text_pipe(x, nlp,allowed_postags=allowed_postag)
        st.write(clean_func(context))


def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text


def chunck_list(lst,chunck_size = 5000):
    '''Splitting large doc into batches, defualt batch size 5000'''
    for i in range(0,len(lst),chunck_size):
        yield lst[i:i+chunck_size]


def clean_text_pipe(text,spacy_pipe,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''Remove stop words and punctuaion'''
    BATCH_SIZE = 5000
    if len(text) > BATCH_SIZE:
        split_text = chunck_list(text)
        docs = [spacy_pipe(t) for t in split_text]
        cleaned_docs = []
        for doc in docs:
            cleaned_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and token.pos_ in allowed_postags]
            cleaned_text=' '.join(cleaned_doc)
            cleaned_docs.append(cleaned_text)
        return combine_texts(cleaned_docs)
    else:
        doc = spacy_pipe(text)
        cleaned_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and token.pos_ in allowed_postags]
        cleaned_text=' '.join(cleaned_doc)
        return cleaned_text



def run_the_model(state):
    st.title('Feature Extraction based on Roberta model')
    with st.spinner('Wait for it...'):
        qa_model = load_roberta_model()
        # list of questions
        # list of questions
    context_region = st.empty()
    selected_col = st.sidebar.selectbox('Corpus from', list(state.df.columns))
    question_asked = st.text_area(label='Question')
    if st.button('Query'):
        with st.spinner('Wait for it...'):
            if question_asked not in get_questions():
                get_questions().append(question_asked)
            query = lambda x: get_answer(context=x, question=question_asked, model=qa_model)
            state.df[question_asked] = state.df[selected_col].apply(query)
    st.write(state.df)