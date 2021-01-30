from gensim import corpora
import gensim
from gensim.models import LdaModel, CoherenceModel,TfidfModel
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict

def cluster_data(state):
    # Bigram model
    data_words_bigrams = make_bigrams(state)
    INPUT = data_words_bigrams
    # Create Dictionary
    id2word = corpora.Dictionary(INPUT)
    # Create Corpus
    texts = INPUT
    # Filter out words that occur less than and greater than
    id2word.filter_extremes(no_below=state.no_below,no_above=state.no_above)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    TOPICS_LIST = range(1, state.lda_topics + 1)
    lda_models = []
    coherence_scores = []
    for TOPICS in TOPICS_LIST:
        lda_model = run_LDA_model(corpus, id2word, TOPICS)
        lda_models.append(lda_model)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                             coherence='c_v')
        score = coherence_model_lda.get_coherence()
        coherence_scores.append(score)

    return coherence_scores, lda_models,corpus

def fit_best_model(chose_topic,lda_models,corpus):
    topic_num = chose_topic-1

    best_lda_model = lda_models[topic_num]

    corpus_trans = best_lda_model[corpus]
    # see which topics they are
    lst = []
    for i in corpus_trans:
        lst.append(i[0][0][0])

    return lst






def run_LDA(state):
    # to Bigrams
    topics = st.sidebar.number_input("# of Topics",min_value=1,max_value=10,value=3,)
    st.title("LDA")
    topic_keywords = st.beta_expander('topics')
    coherence_score = st.beta_expander('Coherence score')
    data_words_bigrams = make_bigrams(state.df_clean, state.text_col_name)
    INPUT = data_words_bigrams
    # Create Dictionary
    id2word = corpora.Dictionary(INPUT)
    # Create Corpus
    texts = INPUT
    # Filter out words that occur less than and greater than
    id2word.filter_extremes(no_below=5)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
    if st.sidebar.button('Try'):
        TOPICS_LIST = range(1,topics+1)
        lda_models = []
        coherence_scores = []
        for TOPICS in TOPICS_LIST:
            lda_model = run_LDA_model(corpus, id2word, TOPICS)
            lda_models.append(lda_model)

            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                                 coherence='c_v')
            score = coherence_model_lda.get_coherence()
            coherence_scores.append(score)
        with coherence_score:
            # fig,ax = plt.subplots()
            # sns.lineplot(x=TOPICS_LIST, y=coherence_scores,ax=ax)
            # plt.title('Coherence_scores')
            # plt.xlabel('TOPICS')
            # plt.ylabel('Scores')
            # st.pyplot(fig)
            df_plot = inspect_term_frequency(corpus,id2word,n=15)
            # fig, ax = plt.subplots(1, 3, figsize=(16, 12))
            # for i in range(1,topics+1):
            #     topic = 'topic' + str(i)
            #     topic_df = train.loc[train['top1'] == topic, 'document']
            #     frequency = inspect_term_frequency(topic_df)
            #     sns.barplot(data=frequency, x='frequency', y=frequency.index, ax=ax[i])
            #     ax[i].set_title(f"Topic {i + 1} - Top words")
            # plt.tight_layout()

        # fig,ax = plt.subplots()
        # sns.lineplot(x=TOPICS_LIST, y=coherence_scores,ax=ax)
        # plt.title('Coherence_scores')
        # plt.xlabel('TOPICS')
        # plt.ylabel('Scores')
        # st.pyplot(fig)
    selected_topics = st.sidebar.text_input('LDA_topics:')
    if st.sidebar.button('Fit'):
        best_lda_model = run_LDA_model(corpus, id2word, int(selected_topics))
        add_probabilities(state.df_clean,state.text_col_name,int(selected_topics),best_lda_model,id2word)
        corpus_trans = best_lda_model[corpus]
        # see which topics they are
        lst = []
        for i in corpus_trans:
            lst.append(i[0][0][0])

        state.df_clean['topic'] = lst
        st.write(state.df_clean.head())

        fig, ax = plt.subplots(1, 3, figsize=(16, 12))
        for i in range(int(selected_topics)):
            topic = str(i)
            topic_df = state.df_clean.loc[state.df_clean['topic'] == topic, state.text_col_name]
            frequency = inspect_term_frequency(corpus,id2word,topic_df)
            sns.barplot(data=frequency, x='frequency', y=frequency.index, ax=ax[i])
            ax[i].set_title(f"Topic {i + 1} - Top words")
        st.pyplot(fig)
# LDA model
def run_LDA_model(corpus, id2word,TOPICS):
    '''This function returns lda models '''
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=TOPICS,
                                           random_state=42,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    return lda_model

def split_text(text ,n=10000):
    '''take text as input and return its first n percentage of elements'''
    if len(text ) >n:
        return text[:n]
    else:
        return text

def tokenize_text(text):
    '''transforming text in to token'''
    token_list = text.split()
    return token_list

def make_bigrams(state):
    # consider some word may occur in bigram format like New York
    pure_text = state.df[state.column].tolist()
    pure_token = list(map(tokenize_text, pure_text))
    bigram = gensim.models.Phrases(pure_token, min_count=int(state.min_count), threshold=int(state.threshold))
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in pure_token]


def cal_probabilities(text,lda,id2word):
    """Add probabilities for topics for a document."""
    # Predict probabilities
    tokens = tokenize_text(text)
    corpus = id2word.doc2bow(tokens)
    predictions = lda.get_document_topics(corpus, minimum_probability=0.0)
    topics = [topic for topic, probability in predictions]
    return [prediction[1] for prediction in predictions]


def add_probabilities(df_clean,select_col,num_topics,lda,id2word):
    """Add prob to dataframe"""
    columns = ['topic' + str(i + 1) for i in range(num_topics)]
    cal_prob = lambda x: cal_probabilities(x,lda,id2word)
    df_clean[columns] = df_clean[select_col].apply(cal_prob).tolist()
    return df_clean


def inspect_term_frequency(corpus,id2word, n=15):
    """Show top n frequent terms in corpus."""

    # Find term frequencies
    frequency = defaultdict(lambda: 0)
    for document in corpus:
        for codeframe, count in document:
            frequency[codeframe] += count
    frequency_list = [(codeframe, count) for codeframe, count in frequency.items()]
    frequency_list.sort(key=lambda x: x[1], reverse=True)
    codeframe_lookup = {value: key for key, value in id2word.token2id.items()}
    data = {codeframe_lookup[codeframe]: count for codeframe, count in frequency_list[:n]}
    return pd.DataFrame(pd.Series(data), columns=['frequency'])


