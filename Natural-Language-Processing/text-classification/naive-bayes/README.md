# The Naive Bayes algorithm

The Naive Bayes Algorithm is a probabilistic ML model that is mainly used for classification task. The idea of the algorithm is based on the Bayes theorem:

 $$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent of each other (i.e. that the presence of one particular feature does not affect the other) Hence it is called naive. Another basic assumption made here is that all the predictors have an equal effect on the outcome.

For classification tasks, we have Multinomial and Bernoulli Naive Bayes. This is mostly used for document classification problem, i.e whether a document belongs to the category of sports, politics, technology etc. The features/predictors used by the classifier are the frequency of the words present in the document. When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution and use a Gaussian Naive Bayes approach. 

Naive Bayes algorithms are mostly used in sentiment analysis, spam filtering, recommendation systems etc. They are fast and easy to implement, i.e. they converge faster and requires less training data. Compared to other discriminative models like logistic regression, the Naive Bayes model it takes lesser time to train. Their biggest disadvantage, however, is the requirement of predictors to be independent. In most of the real life cases, the predictors are dependent, this hinders the performance of the classifier.
