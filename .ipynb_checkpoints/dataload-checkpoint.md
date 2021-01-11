# PDF files Clustering APP

An application that automize the text data preprocessing task in SIA NLP project.

Using Latent Dirichlet Allocation(LDA) technique to cluster files.

# Procedure

1. Import Data
> * 1.Import from folder: using pypdf2 to conduct text extraction, might not be accurate but quite fast.
> * 2.Import from file: read csv/pickle type files that have include text data inside.
2. Data Preparation
> * 1.Including clean puncatuation, remove stop word, and lemmatization.
> * 2.Tokenization( considering bigrams).
3. Topic Clustering
> * 1.Using bag of words to conduct LDA
> * 2.Using Tf-idf to conduct LDA
4. Performance
> * 1.Coherence value
> * 2.Wordcloud
---
**Brief note on LDA**

Suppose you have the following set of sentences:

    I like to eat broccoli and bananas.
    I ate a banana and spinach smoothie for breakfast.
    Chinchillas and kittens are cute.
    My sister adopted a kitten yesterday.
    Look at this cute hamster munching on a piece of broccoli.

What is latent Dirichlet allocation? It’s a way of automatically discovering topics that these sentences contain. For example, given these sentences and asked for 2 topics, LDA might produce something like

* Sentences 1 and 2: 100% Topic A
* Sentences 3 and 4: 100% Topic B
* Sentence 5: 60% Topic A, 40% Topic B
* Topic A: 30% broccoli, 15% bananas, 10% breakfast, 10% munching, … (at which point, you could interpret topic A to be about food)
* Topic B: 20% chinchillas, 20% kittens, 20% cute, 15% hamster, … (at which point, you could interpret topic B to be about cute animals)