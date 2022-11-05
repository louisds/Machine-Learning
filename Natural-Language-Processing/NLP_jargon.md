# 1) Linguistics and NLP tasks

## Lexicon

Lexicon refers to the component of an NLP system that contains information (semantic, grammatical, ...) about individual words or word strings. In practice, a lexical entry will include further information about the roles the word plays, such as feature information - for example, whether a verb is transitive, intransitive, ditransitive, etc., what form the verb takes (e.g. present participle, or past tense, etc.).

## Vocabulary vs. Corpus

The corpus is the collection of texts used to train an NLP model, while the vocabulary is the collection of words used to train an NLP model. Example: BERT is an advanced NLP model trained on the entire content of Wikipedia (originally the English language Wikipedia). The corpus is the collection of Wikipedia articles it was trained on. The vocabulary is the vocabulary of the English language.

## Pragmatics/Semantics/Syntax/Morphology/Phonology/Phonetics

<img src="https://github.com/louisds/NLP-projects/blob/main/images/semantic_pragmatic.png"  width="200">

**Pragmatics** identifies the meaning of words and phrases based on how language is used to communicate (i.e. emphasis on their context as well). Unlike semantics, which only concerns the meaning of the words, pragmatics goes a step further by looking at the same word in relation to its context. Thus, pragmatics explains how language users are able to overcome apparent ambiguity since it explains the meaning relies on the manner, time, place, etc. of an utterance.

**Semantics** is the literal meaning of words and phrases. It has two main categories: lexical semantics and phrasal semantics. Lexical semantics concerns the meanings of words and the meaning of relationships among words, i.e. analyze words and see how they can be related to each other with relations to synonyms, antonyms, homonyms, polysemy, figures of speech. Phrasal semantics concerns the meaning of syntactic units, i.e. concepts such as paraphrase, contradiction, ambiguity, mutual entailment, etc.

For example, the sentence – “He is so cool”. Semantically, this sentence can be interpreted as – He is very nice, a compliment to the person, which is the literal meaning. But under pragmatics, this sentence suggests the context: the positive attitude of the speaker towards the person. This is the intended or the inferred meaning in the sentence.

**Syntax** is the set of rules needed to ensure a sentence is grammatically correct (article, noun, verb, ...). Syntax refers to grammar, while semantics refers to meaning. 

**Morphology** analyzes the structure of words and parts of words such as stems, root words, prefixes, and suffixes (i.e. how words are formed).

**Phonology** concerns the study of more complex and abstract sound patterns and structures (syllables, intonation, etc.), i.e. different patterns of sounds in different positions in words etc.

**Phonetcs** studies how humans produce and perceive sounds. 

## Dependency (Syntactic) vs. Constituency Parsing

Both methods are types of sentence parsing, which is done by taking a sentence and breaking it down into different parts of speech. The words are placed into distinct grammatical categories, and then the grammatical relationships between the words are identified. This creates a parse tree that highlights the syntactical structure of a sentence according to a formal grammar. Both methods use different types of grammars and have different assumptions.

**Dependency parsing** expresses the syntax of the sentence in terms of dependencies between words. The parse tree is a graph G = (V, E) where the set of vertices V contains the words in the sentence, and each edge in E connects two words. Each edge in E has a type, which defines the grammatical relation that occurs between the two words.

**Constituency Parsing** is based on the formalism of context-free grammars. In this type of tree, the sentence is divided into constituents, that is, sub-phrases that belong to a specific category in the grammar. In English, for example, the phrases “a dog”, “a computer on the table” and “the nice sunset” are all noun phrases, while “eat a pizza” and “go to the beach” are verb phrases. This a difference with dependency parsing, which looks at the individual words.

## Lexical-Semantic vs. Structural-Syntactic vs. Pragmatic Ambiguity

**Lexical or Semantic ambiguity** occurs when a word has more than one meaning. Example: “I went to the bank”. There are several dictionary meanings for the word “bank” Here, you can’t know if “bank” is a place where money is held or a riverbank. Lexical ambiguity is a subtype of semantic ambiguity where a word or morpheme is ambiguous instead of an expression.

**Structural or Syntactic ambiguity** arises when the sentence’s meaning is unclear because of how words are related. Example: “The chicken is ready to eat”. You can’t tell if the chicken will do the eating or if the chicken will be eaten. 

**Pragmatic ambiguity** arises when the statement is not specific, and the context does not provide the information needed to clarify the statement. It deals with the use of real-world knowledge and understanding how this impacts the meaning of what is being communicated. Example: "Do you know what time it is?". This could just be someone asking for the time, or someone being angry because other person was too late.

## Taxonomy and Ontology

Taxonomy and ontology work in different ways to apply structure to language. 

**Taxonomy** identifies the hierarchical relationships among concepts and specifies the term to be used to refer to each, i.e. it prescribes structure and terminology. A taxonomy is static and classifies words into categories and sub-categories (i.e. chicken is a bird which is an animal which is a living organism on earth).

**Ontology** identifies and distinguishes concepts and their relationships, i.e. it describes content and relationships. An ontology is dynamic and domain-centric and provides types, properties, and inter-relationships of the words.

Together, they help disambiguate language so programs can perform with more accuracy. In the example “The chicken is ready to eat”, the taxonomy can classify the chicken as an animal, while ontology can apply relational understanding to further define the term “chicken” as either an animal or a type of food to be eaten.

## Soundex

Soundex converts an alphanumeric string to a four-character code that is based on how the string sounds when spoken in English.

## Information Extraction vs. Retrieval

**Information Extraction (IE)** is the the process of sifting through unstructured data and extracting vital information into more editable and structured data forms. Examples are summaries from vast collections of text like Wikipedia, conversational AI systems like chatbots, extracting stock market announcements from financial news, sentiment analysis, topic modeling, NER, etc.

**Information Retrieval (IR)** searches a collection of natural language documents with the goal of retrieving exactly the set of documents that matches a user’s question (i.e. query). Think of library systems, google search, etc. 

## Named Entity

In information extraction, a named entity is a real-world object, such as a person, location, organization, product, etc., that can be denoted with a proper name. It can be abstract or have a physical existence. Examples of named entities include Barack Obama (person), New York City (city), Volkswagen Golf (car), or anything else that can be named. Named entities can simply be viewed as entity instances (e.g., New York City is an instance of a city).

## Named Entity Recognition (NER)

Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, etc. Example:

$Jim_{Person}$ bought 300 shares of $Acme \ Corp_{Organization}$ in $2006_{Time}$.

In this example, a person name consisting of one token, a two-token company name and a temporal expression have been detected and classified.

## Named Entity Congruence

The contextual relationship between candidates for named entities in the same sentence. 

<img src="https://github.com/louisds/NLP-projects/blob/main/images/congruence.png"  width="500">

## Named Entity Recognition and Disambiguation (NERD)

Named Entity Recognition and Disambiguation (NERD) or Named Entity Linking (NEL) is the task of recognizing (i.e. NER) and disambiguating (i.e. NED) named entities to a knowledge base (e.g. Wikidata, DBpedia, or YAGO). It can be split in two classes of approaches

**End-to-End:** processing a piece of text to extract the entities (i.e. Named Entity Recognition) and then disambiguate these extracted entities to the correct entry in a given knowledge base (e.g. Wikidata, DBpedia, YAGO).

**Disambiguation-Only:** contrary to the first approach, this one directly takes gold standard named entities as input and only disambiguates them to the correct entry in a given knowledge base.

Example for Wikipedia:

Barack Obama was born in Hawaï.

Barack -> https://en.wikipedia.org/wiki/Barack_Obama

Obama -> https://en.wikipedia.org/wiki/Barack_Obama

Hawaï -> https://en.wikipedia.org/wiki/Hawaii

## Relation Extraction

Relationship extraction is the task of extracting semantic relationships from a text. Extracted relationships usually occur between two or more entities of a certain type (e.g. Person, Organisation, Location) and fall into a number of semantic categories (e.g. married to, employed by, lives in). For example in "Louis loves Dasha", we could extract the relationship Louis -> Married with -> Dasha. 

## Sentence Order Prediction (SOP)

Sentence order prediction is the task of finding the correct order of sentences in a randomly ordered document. Correctly ordering the sentences requires an understanding of coherence with respect to the chronological sequence of events described in the text. It can be introduced as a self-supervised loss (used in ALBERT), where it primary focuses on inter-sentence coherence and is designed to address the ineffectiveness of the next sentence prediction (NSP) loss (proposed in the original BERT).

## Co-reference Resolution (CRS)

Co-reference resolution is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction. Example:

"I voted for Bert because he was most aligned with my values" - she said.

CRS will link I, my, and she, with the same person, and Bert and he.


## POS tagging

Part of Speech (POS) is a  category to which a word is assigned in accordance with its syntactic functions. In English the main parts of speech are noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection. POS tagging is the process of assigning a specific tag to a word in our corpus since the POS tags are used to describe the lexical terms that we have within our text. POS tags describe the characteristic structure of lexical terms within a sentence or text, therefore, we can use them for making assumptions about semantics. Other applications of POS tagging include: Named Entity Recognition, Co-reference Resolution, Speech Recognition, etc.

## Tokenization

Tokenization is the process of breaking a stream of textual data into words, terms, sentences, symbols, or some other meaningful elements called tokens. This is necessary to create an input format for an NLP model. Tokenization might seem simple (e.g. just stringsplit with spaces to create a list of words), but there are different ways of tokenization (e.g. taking New York as one token instead of two, etc.). In short: splitting a text or documents in tokens, which are the basic meaningful unit of language. 

## Morpheme

Morpheme is defined as a base form of the word. A token is basically made up of two components: one is morphemes and the other is inflectional formlike prefix or suffix. For example, consider the word Antinationalist (Anti + national+ ist ) which is made up of Anti and ist as inflectional forms and national as the morpheme.

## Keyword Normalization

Normalization is the process of converting a token into its base form. In the normalization process, the inflectional form of a word is removed so that the base form can be obtained. Example: the normal form of "antinationalist" is national. Normalization is helpful in reducing the number of unique tokens present in the text, removing the variations in a text, and also cleaning the text by removing redundant information. Two popular methods used for normalization are stemming and lemmatization.

## Stemming vs. Lemmatization

**Stemming** usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes.

**Lemmatization** usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

Example: If confronted with the token "saw", stemming might return just s, whereas lemmatization would attempt to return either see or saw depending on whether the use of the token was as a verb or a noun. 

# 2) NLP Basics

## Language Model

In many Natural Language tasks you have a language (L) and want to build a model (M) for the language. In NLP we think of language L as a process for generating text, where the process is a collection of probability distributions. Given a history $h$ consisting of a series of previous words in a sentence, the language $L$ is the probability that the next word is $w$. Similarly, given an entire sentence $s$, we can evaluate $L(s)$ as the probability of the sentence occurring. A language model M will try to replicate this language, which we can then use for various tasks. 

$$
p(w_1, ..., w_n) = p(w_1) \cdot p(w_2|w_1) ... p(w_n|w_1, ..., w_{n-1})
$$

## Source Channel Framework
<img src="https://github.com/louisds/NLP-projects/blob/main/images/noisy_channel_model.png"  width="500">

The source channel framework or noisy channel framework is a framerwork used in spell checking, speech recognition, translation, POS tagging, etc. It consists of a message (input X) that is encoded and fed into a (noisy) channel. The output from the noisy channel (output Y) is then decoded (decoded message X'), attempting to reconstruct the original message based on this noisy output. In NLP we do not usually act on encoding. The problem is reduced to decoding the noisy output Y for getting the most likely input given the output. 

An example of a noisy channel for spell checking is a channel that scrambles some letters in an input sentence (and thus producing spelling errors). In this case, the correct language (real language) is the input X, while the output Y is the language with errors (observed language). The probability P(X) is the probability of the word sequence in the correct language (language model probability). The probability P(Y|X) is the probability that we have a specific scrambled output sentence Y, giving a specific input sentence X (noisy channel probability). The goal of the framework is the obtain X', based on the probability P(X|Y): 

$$
X' = argmax_x \ P(x|Y) = argmax_x \ \frac{p(x) p(Y|x)}{P(Y)} \propto argmax_x \ p(w) p(Y|x)
$$

## N-grams

If we have a sequence of $n$ words, the number of histories (consisting of a series of previous words) grows as $|V|^{n-1}$, while the number of parameters in our model grows as $|V|^{n}$, where V is our vocabulary. To reduce the number of parameters, we group histories with a particular grouping method. N-gram language modeling is a popular grouping method, where the probability of a given sequence of N words is predicted. For example in case of a trigram (3-gram), we ignore everything except for the previous two words when we try to predict the next word (number of parameters is of the order $|V|^{3}$).

Despite the fact that a higher-order n-gram model, in theory, contains more information about a word's context, it cannot easily generalize to other data sets (i.e. overfitting) because the number of events (i.e. n-grams) it has seen during training becomes progressively less as n increases. On the other hand, a lower-order model lacks contextual information and so may underfit your data.

## Perplexity

We want our language model $M$ to assign high probabilities to sentences that are real and synthetically correct. The best model is the one that assigns the highest probability to the test set. The perplexity is a measure to quantify how "good" a language model is, based on a test (or validation) set. The perplexity on a sequence $s$ of words is defined as:

$$
Perplexity(M) = M(s)^{(-1/n)} = \left( p(w_1, ..., w_n)\right)^{(-1/n)}
$$

The intuition behind this metric is that if a model assigns a high probability to a test set, it is not surprised to see it (not perplexed by it), which means the model M has a good understanding of how the language L works. Hence, a good model has a lower perplexity. The exponent (-1/n) in the formula is just a normalizing strategy (geometric average), because adding more sentences to a test set would otherwise introduce more uncertainty (i.e. larger test sets would have lower probability). So by introducing the geometric average, we have a metric that is independent of the size of the test set. 

As an example: a perplexity of 100 means that, on average, the model predicts as if there were 100 equally likely words to follow. In theory, the optimal situation would be a perplexity of 1, where the model always knows exactly which word to predict. However, this means there could only be one possible sentence in a language, which is quite boring. The generated text is either repetitive or incoherent. Overall, it is hard to provide a benchmark for perplexity, as this depends on the vocabulary size and also the fact that we rather stay closer to human perplexity (instead of 1). 

## Pointwise Mutual Information (PMI)

PMI is a metric that quantifies how likely specific words/clusters are to co-occur together within some window, compared to if they were independent. It helps to understand whether two (or more) words actually form a unique concept. If this is the case, we can reduce the dimensionality of our task, since that couple of words (called bigram- or n-grams in the general case for n words) can be considered as a single word. Hence, we can remove one vector from our computations. The bigram "New York" is an example: the two words form a whole, but "new" probably also appears in a lot of other parts within the text. The idea of PMI is that we want to quantify the likelihood of co-occurrence of two words, taking into account the fact that it might be caused by the frequency of the single words. It is defined as:

$$
PMI(w_1, w_2) = log\left( \frac{p(w_1, w_2)}{p(w_1) p(w_2)}  \right) 
$$

So PMI computes the (log) probability of co-occurrence scaled by the product of the single probability of occurrence. When the ratio equals 1 (hence the log equals 0), it means that the two words together don’t form a unique concept: they co-occur by chance. On the other hand, if either one of the words (or even both of them) has a low probability of occurrence if singularly considered, but its joint probability together with the other word is high, it means that the two are likely to express a unique concept.

# Word Embeddings

A word embedding is a learned representation for text where words that have the same meaning have a similar representation. Individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values can for example be learned in a way that resembles a neural network. Each word is represented by a real-valued vector, often tens or hundreds of dimensions. This is contrasted to the thousands or millions of dimensions required for sparse word representations, such as a one-hot encoding. Examples are Word2Vec, GloVe, BERT, etc.

## Cosine Similarity

Cosine similarity is one of the metrics to measure the similarity between two words (or even two documents, irrespective of their size). Mathematically, Cosine similarity metric measures the cosine of the angle between two n-dimensional vectors projected in a multi-dimensional space:

$$
cos-similarity(v_1, v_2) = \frac{v_1 \cdot v_2}{\lVert v_1 \lVert \ \lVert v_2 \lVert}
$$

The Cosine similarity of two documents will range from 0 to 1. If the Cosine similarity score is 1, it means two vectors have the same orientation. The value closer to 0 indicates that the two documents have less similarity. Hence, it is thus a judgment of orientation and not magnitude. This makes it a better metric than Euclidian distance for word similarity.

## Jaccard Similarity

Jaccard Similarity (also known as the Jaccard index and Intersection over Union) is used to determine the similarity between two text document,. This is done by measuring how close the two text documents are to each other in terms of their context, i.e. how many common words are exist over total words. Jaccard Similarity defined as an intersection of two documents divided by the union of that two documents that refer to the number of common words over a total number of words:

$$
J(doc_1, doc_2) = \frac{doc_1 \cap doc_2}{doc_1 \cup doc_2}
$$

Here, we will use the set of words to find the intersection and union of the document. The Jaccard Similarity score is in a range of 0 to 1. If the two documents are identical, Jaccard Similarity is 1. The Jaccard similarity score is 0 if there are no common words between two documents.

## Term Frequency (TF)

Term frequency (TF) is how often a term $t$ (i.e. a word) appears in a document $d$, divided by how many words there are. So it is a measure of how common a word is.

$$
TF(t, d) = \frac{count(t, d)}{|d|}
$$

Consider a document containing 100 words wherein the word apple appears 5 times. The term frequency (i.e., TF) for apple is then (5 / 100) = 0.05.

## Inverse Document Frequency (IDF)

Inverse document frequency (IDF) measures how unique or rare a word $t$ is. It is defined as the natural logarithm of Total number of documents in our corpus $D$, divided by the number of documents with term $t$ in it.

$$
IDF(t, D) = ln \left(\frac{|D|}{|\{ d \in D: \ t \in d \} |} \right)
$$

It is sometimes defined slightly different (e.g. with a +1 in the numerator and denominator to prevent undefined errors). Now, assume we have 10M documents and the word apple appears in 1K of these. Then, the IDF is calculated as ln(10,000,000 / 1,000) = 9.21. 

## Term Frequency–Inverse Document Frequency (TF-IDF)

TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document $d$ in a corpus $D$. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. In other words, TF-IDF is a measure of originality of a word by comparing the number of times a word appears in a document $d$ with the number of documents the word appears in. It is simply the product of TF and IDF.

By combining TF with IDF, we have the weight of how much a word occurs in a document (TF), while at the same time this weight is being diminished for terms that occur very frequently in the corpus (i.e. document set) and being increased for terms that occur rarely. A good example would be "the". This word will probably have a high TF, but a low IDF, which means that the weight will decrease if we calculate TF-IDF. 

## Bag of words (BOW)

BoW is used to extract feature sets from text during the data pre-processing phase, i.e. to convert a text document into numbers that can be used by a model. BOW models encode every word in the vocabulary as one-hot-encoded vector i.e. for vocabulary of size |V|, each word is represented by a |V| dimensional sparse vector with 1 at index corresponding to the word and 0 at every other index. In the weighted variation, it is a weighted sum according to frequency or TF-IDF scores. As vocabulary may potentially run into millions, bag of word models face scalability challenges. It is also not able to capture context.

## Continuous Bag of words (CBOW) vs. Skip gram

Both models will generate better feature (i.e. dense) vectors by using the context of a word and a Neural Network.

**CBOW** used a Neural Network that tries to predict the current word from a window of surrounding context words. The order of the context words does not have an influence on the prediction. The CBOW model is several times faster to train than the skip-gram model, and it has slightly better accuracy for the frequent words.

**Skip Gram** uses a Neural Network that tries to predict the surrounding window of context words, based on the current word. It weighs nearby context words more heavily than more distant words. Skip gram works well with small amounts of the training data, and is generallu better at representing rare words or phrases.

## Word Co-occurence Matrix

The co-occurrence matrix will have specific entities in rows (ER) and columns (EC). The purpose of this matrix is to present the number of times each ER appears in the same context as each EC. As a consequence, in order to use a co-occurrence matrix, you have to define your entites and the context in which they co-occur. Example: the matrix indicates how many times the row word (e.g. 'digital') is surrounded (in a sentence, or in the n word window - depends on the application and your choice of n) by the column word (e.g. 'pie'). Note that the co-occurrence matrix is always symmetric.


## Negative Sampling

A technique used to modify (or even optimize) the CBOW and Skip gram training. Instead of trying to predict the probability of being a nearby word for all the words in the vocabulary, we try to predict the probability that our training sample words are neighbors or not. So instead of having one giant softmax — classifying among |V| classes, we have now turned it into |V| binary classification problem.

## Latent Semantic Indexing (LSI) (a.k.a. Analysis (LSA))

## Latent Dirichlet Allocation (LDA)

# 3) Advanced NLP and Deep Learning

## Masked Language Model

A masked image modell masks words from a sequence of sentences and the designed model needs to predict the masked words to complete the sentence. 

## Ensemble Method

Ensemble methods are approaches to reduce the variance of neural network models. This is done by training multiple models instead of a single model and by combining the predictions from these models. This is called ensemble learning and not only reduces the variance of predictions but also can result in predictions that are better than any single model. Combining the predictions from multiple neural networks adds a bias that in turn counters the variance of a single trained neural network model.

## Transfer Learning (TL)

Transfer learning is a technique where we have a pre-trained deep learning model that is trained on a large dataset (e.g. Wikipedia), which we fine-tune to perform similar tasks on our own dataset.

## Multi Task Learning (MTL)

Multi-task learning (MTL) is a field of machine learning in which models using data from multiple tasks are trained at the same time. This is done using shared representations to uncover the common ideas among a group of tasks that are connected. These shared representations help to overcome the well-known drawbacks of deep learning by increasing data efficiency. The two most common methods for performing MTL with deep neural networks are hard and soft parameter sharing.

**Hard parameter sharing** is done by sharing the hidden layers amongst all jobs while at the same time maintaining several task-specific output layers. This is a very beneficial strategy since it reduces the danger of overfitting by having a model for diverse tasks using a common hidden layer.

**Soft parameter sharing** adds a constraint to achieve similarity among related parameters rather than sharing the same value. Furthermore, we penalize the difference in parameters across the models that we train for each task. By loosely connecting the shared space representations, this method, in contrast to rigid sharing, allows activities greater flexibility.

The main mechanisms of MTL are: Attention focusing, regularization, and implicit data augmentation. 

## Meta-Learning

Meta-learning is a field in machine learning studying approaches to learn better learning algorithms. Approaches aim at improving algorithms in various aspects, including data efficiency and generalizability. 

## Continual Learning

Continual learning, also known as lifelong learning or online machine learning is a fundamental concept in machine learning (ML), in which the models continuously learn and evolve based on the input of increasing amounts of data while retaining previously-learned knowledge.

## Feature-Extraction vs. Fine Tuning

These two types of transfer learning. In **finetuning**, we start with a pretrained model and update all of the model’s parameters for our new task, in essence retraining the whole model. In **feature extraction**, we start with a pretrained model and only update the final layer weights from which we derive predictions. It is called feature extraction because we use the pretrained NN as a fixed feature-extractor, and only change the output layer.

## Special Tokens (CLS, SEP, MASK)

These are special tokens that convey specific meaning to the encoder:

**CLS** stands for classification. It is added to represent sentence-level classification. It is generally added at the start of a sequence so that it represents the entire sentence.

**SEP** is used during the next sentence prediction. It is a delimiter token that helps the model know which is the next sentence. In the case of a single sentence, it is just added at the end.

**MASK** is used during the Masked Language Modeling phase. It is used to represent the masked word to the model.

## Token, Segment/Sentence and Position embedding

These are three forms of embeddings:

**Token embeddings** are needed to identify the word/subword being processed as input, as well as the token being masked. They are just the pre-trained embeddings for different words. In BERT, each word piece token will be converted into a 768-dimensional vector representation by the Token Embeddings layer.

**Sentence or Segment embeddings** are needed for next sentence prediction (NSP). They are needed to easily tell apart the different parts of the input and are basically the sentence number that is encoded into a vector. The model must know whether a particular token belongs to sentence A or sentence B.

**Positional embeddings** are needed to distinguish the same token in different positions. They are learned vectors for every possible position between 0 and the sequence length, indicating the position of the word within the sentence that is encoded into a vector. BERT was created to handle input sequences up to 512 characters long. This means that the Position Embeddings layer in BERT is a lookup table with a size of (512, 768) rows, with the first row representing the vector representation of any word in the first position, the second row representing any word in the second position, and so on.

For example in BERT models, we sum up these three types of embedding, because these are the pieces of information needed to accomplish the loss' tasks, which is, both masked language modeling (i.e. predicting the masked tokens) and next sentence prediction (i.e. predict whether the second segment followed the first segment in the original text). A normal transformer architecture already adds token embeddings and positional embeddings (the latter because TFs don't have a sequential nature as recurrent neural networks, so some information about the order of the input is needed - if you disregard this, our output will be permutation-invariant). 

## Encoder-Decoder model (seq2seq)

Encoder-Decoder model consists of two RNNs (e.g. GRU, LSTM, etc.) and are very popular in translation. It encodes the input sequence to one fixed length vector from which to decode the output at each time step. An issue with this structure is the information bottleneck layer, where the whole input sequence is just represented as one vector. This issue is believed to a problem when decoding long sequences because it make difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus. Sometimes we only need an encoder, or only need a decoder. A summary of all the versions and their applications:

**Encoder-only models:** Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.

**Decoder-only models:** Good for generative tasks such as text generation.

**Encoder-decoder models or Seq2Seq models:** Good for generative tasks that require an input, such as translation or summarization.

## Attention mechanism

Attention is a mechanism that solves the bottleneck layer problem that arises in orginal seq2seq RNN models. On each step of the decoding, the attention mechanism directly connects to the encoder and focuses on a particular part of the input sequence. We concatenate the attention output with the decoder hidden state to predict the next word in the decoder sequence. As opposed to normal attention (where Y pays attention to X), there is also self-attention (where X pays attention to X):

**Self-Attention** is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. This allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores. This can for example be used in an encoder. A decoder can use both self-attention and normal attention.

Next to single-head attention, we also have multi-head attention. Multi-head attention runs through several attention heads with different parameters in parallel. The independent attention outputs are then concatenated and linearly transformed into the expected dimension

## Transformer

A Transformer is an architecture that aims to solve seq2seq tasks while handling long-range dependencies with ease. It relies entirely on the attention mechanism to compute representations of its input and output without using sequence-aligned RNNs. Originally, they consisted out of a encoder-decoder model, but we can also have encoder-transformers and decoder-transformers:

**Encoder models** use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called auto-encoding models. The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence. That is why they are often called auto-encoding models, as these kind of language model aims to reconstruct the original data from corrupted input.

Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering. Representatives of this family of models include: ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa.

**Decoder models** use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called auto-regressive models, as they use the model outputs of previous time steps for the new prediction. The pretraining of decoder models usually revolves around predicting the next word in the sentence. These models are best suited for tasks involving text generation. Representatives of this family of models include: CTRL, GPT, GPT-2, Transformer XL. 

**Encoder-decoder models** (also called seq2seq models) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input. The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces. Seq2seq models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering. Representatives of this family of models include: BART, mBART, Marian, T5.


## Permutation-Based Language Modelling

Permutation-Based Language Modelling is a type of language modeling, used by XLNet to outperform BERT models. What made BERT stand out among all traditional language models, was its ability to capture the bidirectional context. However, its major flaw was introducing [MASK] tokens and the assumption that the predicted (masked) tokens are independent of each other given the unmasked tokens (i.e. parallel independent predictions in pre-training). 

Permutation-Based Language Modelling builds a model that incorporates bidirectional context while avoiding the [MASK] token and parallel independent predictions. They are trained to predict one token given preceding context like a traditional language model, but instead of predicting the tokens in sequential order, it predicts tokens in some random order. To make this clear, let’s take the following sentence as an example:

“Sometimes you have to be your own hero.”

A traditional language model would predict the tokens in the order

“Sometimes”, “you”, “have”, “to”, “be”, “your”, “own”, “hero”

where each token uses all previous tokens as context.

In permutation language modeling, the order of prediction is not necessarily left to right. For instance, it could be

“own”, “Sometimes”, “to”, “be”, “your”, “hero”, “you”, “have”

where “Sometimes” would be conditioned on seeing “own “, “to” would be conditioned on seeing “own” & “Sometimes” and so on.

## Embedding Factorization

Embedding matrices consume a lot of memory which poses a challenge for edge deployment. They typically contain most of the parameters for language models and about a third for machine translation systems. Embedding factorization is a compression method to reduce the amount of memory needed. Example technique: Distilled Embedding,which is based on low-rank matrix decomposition and knowledge distillation.

# 4) Model Training and Evaluation

## Precision

Precision is an indicator of a machine learning model’s performance and reflects the quality of a positive prediction made by the model. Precision refers to the number of true positives divided by the total number of positive predictions:

$$
PRECISION = \frac{TP}{TP + FP}
$$

## Recall/Sensitivity (True Positive Rate)

The recall measures the model's ability to detect positive samples. The higher the recall, the more positive samples detected. It is calculated as the ratio between the numbers of positive samples correctly classified to the total number of positive samples:

$$
RECALL = \frac{TP}{TP + FN}
$$

## Specificity (True Negative Rate)

Specificity tells us what proportion of the negative class got correctly classified:

$$
SPECIFICITY = \frac{TN}{TN + FP}
$$


## F1 score

F1-score is one of the most important evaluation metrics in classification. It elegantly sums up the predictive performance of a model by combining two otherwise competing metrics — precision and recall. It is defined as follows:

$$
F1 = 2 \frac{PRECISION \cdot RECALL}{PRECISION + RECALL}
$$

Which is essentially taking the harmonic mean of precision and recall.

## Receiver Operator Characteristic (ROC) Curve 

The ROC curve is an evaluation metric for binary classification problems. It is a probability curve that plots the TPR (TP/TP+FN) against the FPR (FP/FP+TN) at various threshold values and essentially separates the ‘signal’ from the ‘noise’.

## Area Under the Curve (AUC)

The AUC is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.

When AUC = 1, then the classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly. If, however, the AUC had been 0, then the classifier would be predicting all Negatives as Positives, and all Positives as Negatives. When 0.5 < AUC < 1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.

When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points. Meaning either the classifier is predicting random class or constant class for all the data points.

## BLUE Score

BLUE (Bi-Lingual Evaluation Understudy) is a popular evaluation metric for Machine Translation. In short, BLEU compares the machine’s translation (known as the candidate translation) with existing human-generated translations (known as the reference translations). BLEU works by computing the precision, i.e. the fraction of tokens from the candidate that appear (or are “covered”) by the references, while also penalizing words that appear in the candidate more times than it appears in any of the references. 

## ROUGE

ROUGE is a set of evaluation metrics, rather than just one. We will cover the main ones that are most likely to be used:

**ROUGE-N** measures the number of matching ‘n-grams’ between our model-generated text and a reference text. The reference is a human-made best-case output — so for automated summarization is would be a human-made summary of our input text. For machine translation, it would be a professional translation of our input text.

**ROUGE-Recall** counts the number of overlapping n-grams found in both the model output and reference text and then divides this number by the total number of n-grams in the reference. This is great for ensuring our model is capturing all of the information contained in the reference but this isn’t so great at ensuring our model isn’t just pushing out a huge number of words to game the recall score.

**ROUGE-Precision** counts the number of overlapping n-grams found in both the model output and reference text and then divides this number by the total number of n-grams in the model output. This makes sure that our model isn't just pushing out a huge number of words.

**ROUGE-F1-Score** gives us a reliable measure of our model performance that relies not only on the model capturing as many words as possible (recall) but doing so without outputting irrelevant words (precision).

**ROUGE-L** measures the longest common subsequence (LCS) between our model output and reference. All this means is that we count the longest sequence of tokens that is shared between both. The idea here is that a longer shared sequence would indicate more similarity between the two sequences. We can apply our recall and precision calculations just like before — but this time we replace the match with LCS.

**ROUGE-S** or skip-gram concurrence metric allows us to search for consecutive words from the reference text, that appear in the model output but are separated by one-or-more other words.

ROUGE is a great evaluation metric but comes with some drawbacks. In particular, ROUGE does not cater for different words that have the same meaning, as it measures syntactical matches rather than semantics.

## Word Error Rate (WER)

Word error rate (WER) is a common metric of the performance of a speech recognition or machine translation system. The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one). The WER solves this by using metrics derived from the Levenshtein distance by working at the word level (number of substitutions S, deletions D, insertions I for the words to get from candidate to reference sentence):

$$
WER = \frac{S + D + I}{S + D + C}
$$


Where C is the number of correct words and (S+D+C) is the number of words in the reference. 

## Scheduling Learning Rate

## Label Smoothening

When using deep learning models for classification tasks, we usually encounter the following problems: overfitting, and overconfidence. Overfitting is well studied and can be tackled with early stopping, dropout, weight regularization etc. On the other hand, we have less tools to tackle overconfidence. Label smoothing is a regularization technique that addresses both problems.

Label smoothing replaces one-hot encoded label vector $y_{hot}$ with a mixture of $y_{hot}$ and the uniform distribution. Label smoothing is used when the loss function is cross entropy, and the model applies the softmax function to the penultimate layer’s logit vectors z to compute its output probabilities p. Original one-hot encoded labels encourages largest possible logit gaps to be fed into the softmax function. Intuitively, large logit gaps combined with the bounded gradient will make the model less adaptive and too confident about its predictions. In contrast, smoothed labels encourages small logit gaps, which restrains the largest logit from becoming much bigger than the rest. This results in better model calibration and prevents overconfident predictions.
