# Ensemble Learning and Random Forests

https://climbtheladder.com/random-forest-interview-questions/

https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-random-forests/

https://iq.opengenus.org/questions-on-random-forest/

https://krishnaik.in/2022/03/01/important-interview-questions-on-random-forest-machine-learning-algorithm/

https://medium.com/@penggongting/in-interview-how-to-answer-compare-random-forest-and-gradient-boosting-decision-tree-105de35cff3b

https://www.linkedin.com/pulse/top-interview-question-machine-learning-decision-tree-deepak-chaubey/

https://www.mlstack.cafe/blog/random-forest-interview-questions

If we aggregate the predictions of a group of predictors (such as classifiers or regressors), we will often get better predictions than with the best individual predictor. A group of predictors is called an ensemble, and an ensemble learning algorithm is called an ensemble method.

# Voting Classifiers

A very simple way to create an even better classifier is to aggregate the predictions of diverse classifiers (Logistic Regression, SVM, Random Forest, etc.) and choose the class that gets the most votes. This majority-vote classifier is called a `hard voting`classifier. Surpringly, the hard voting classifier often gets a higher accuracy than the best classifier in the ensemble. In fact, even if each classifier is a `weak learner`(meaning it does slightly better than random guessing), the ensemble can still be a `strong learner`, provided there are a sufficient number of weak learners and they are sufficiently diverse. This is due to the law of large numbers (think about a biased coin that has a 51% chance of coming up heads). However, this is only true if all classifiers are perfectly independent, making uncorrelated errors. This is hard to achieve, because we train on the same data. A possible solution is to use very different algorithms.
