# Understanding Automatic Text Summarization

## 1) Introduction

Text summarization is the process of shortening a long piece of text with the goal of keeping its meaning and effect as intact as possible. There are mainly four types of summaries:

1. Single Document Summary: Summary of a Single Document
2. Multi-Document Summary: Summary from multiple documents
3. Query Focused Summary: Summary of a specific query
4. Informative Summary: It includes a summary of the full information

There are also two main approaches to perform text summarization:

1. Extraction-based Summarization
2. Abstraction-based Summarization

**Extraction-based Summarization** picks up the most important sentences and lines from the documents and combines them to create a summary. So every sentence in the summary actually belongs to the original text. This means extractive summarizations don’t require natural language generations and semantic representations (which makes them less complicated).

**Abstraction-based Summarization** uses new phrases and terms, different from the actual document, keeping the points the same (just like how we actually summarize). This approach is mainly based on deep learning, which makes it more complex than the extraction-based approach. 

Lastly, we generally use two types of evaluation methods:

1. Human evaluation
2. Automatic evaluation

Human evaluation speaks for itself. Automatic evaluation, however, is a little bit more complicated. A popular method used in this category is ROUGE (Recall-Oriented Understudy for Gisting Evaluation), where the quality of the summary is determined by comparing it to other summaries made by humans as a reference. The intuition behind this is if a model creates a good summary, then it must have common overlapping portions with the human references. Common versions of ROUGE are: ROUGE-n, ROUGE-L, and ROUGE-SU.

## 2) Approaches to Text Summarization

### 2.1) Extraction-based Summarization

The Extractive Summarizers first create an intermediate representation that has the main task of highlighting or taking out the most important information of the text to be summarized based on the representations. There are two main types of representations: Topic representations and Indicator representations.

**Topic Representations** focuses on representing the topics represented in the texts. To get these representations, we could use frequency driven approaches (word probability, TFIDF), where we assign weights to the word based on whether they belong to a topic. Another method is the topic word approach, where word frequencies and a frequency threshold are used to find the word that can potentially describe a topic (classifies the importance of a sentence as the function of the number of topic words it contains). 

**Indicator Representations** depend on the features of the sentences and rank them on the basis of these features. So, here the importance of the sentence is not dependent on the words it contains (as in Topic representations) but directly on the sentence features. Popular methods are Graph-Based methods (text documents as connected graphs) and classic ML methods (classify sentences into summary or non-summary). 

After obtaining the intermediate representations, we move to assign some scores to each sentence to specify their importance. For topic representations, a score to a sentence depends on the topic words it contains, and for an indicator representation, the score depends on the features of the sentences. Finally, the sentences having top scores, are picked and used to generate a summary.

### 2.2) Abstract-based Summarization

Very similar to what humans do to summarize: create a semantic representation of the document in our brains, and then pick words from our general vocabulary (the words we commonly use) that fit in the semantics, to create a short summary that represents all the points of the actual document. Sequence to sequence models with the attention mechanism is a popular approach for this type of summarization.




