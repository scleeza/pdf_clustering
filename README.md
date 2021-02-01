# PDF Clustering

## 1. Import Data ☑️
- Import from folder: using pypdf2 to conduct text extraction, might not be accurate but quite fast.
- Import from file: read csv/pickle type files that have include text data inside.
- [code](https://github.com/scleeza/pdf_clustering/blob/master/page_dataload.py)   
## 2. Data Preparation ☑️
- Including clean puncatuation, remove stop word, and lemmatization.
- Tokenization(considering bigrams).
- Using spaCy to do POS filteration.(e.g. NOUN, ADJ, VERB...etc)
- [code](https://github.com/scleeza/pdf_clustering/blob/master/page_text_clean.py)
## 3. Topic Clustering ☑️ 
- 1.Using bag of words to conduct LDA
- 2.Using Tf-idf to conduct LDA
- [code](https://github.com/scleeza/pdf_clustering/blob/master/page_LDA.py)
## 4. Performance 🚧 
- 1.Coherence value
- 2.Wordcloud
- [code](https://github.com/scleeza/pdf_clustering/blob/master/page_show_data.py)

## Others 🚧 
- OCR 

## Notebooks
- [notebooks](https://github.com/scleeza/NLP_notebooks)
