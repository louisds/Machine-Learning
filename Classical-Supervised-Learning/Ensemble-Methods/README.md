# Ensemble Learning and Random Forests

https://climbtheladder.com/random-forest-interview-questions/

https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-random-forests/

https://iq.opengenus.org/questions-on-random-forest/

https://krishnaik.in/2022/03/01/important-interview-questions-on-random-forest-machine-learning-algorithm/

https://medium.com/@penggongting/in-interview-how-to-answer-compare-random-forest-and-gradient-boosting-decision-tree-105de35cff3b

https://www.linkedin.com/pulse/top-interview-question-machine-learning-decision-tree-deepak-chaubey/

https://www.mlstack.cafe/blog/random-forest-interview-questions

If we aggregate the predictions of a group of predictors (such as classifiers or regressors), we will often get better predictions than with the best individual predictor. A group of predictors is called an `ensemble`, and an ensemble learning algorithm is called an ensemble method.

## Voting Classifiers

A very simple way to create an even better classifier is to aggregate the predictions of diverse classifiers (Logistic Regression, SVM, Random Forest, etc.) and choose the class that gets the most votes. This majority-vote classifier is called a `hard voting`classifier. Surpringly, the hard voting classifier often gets a higher accuracy than the best classifier in the ensemble. In fact, even if each classifier is a `weak learner`(meaning it does slightly better than random guessing), the ensemble can still be a `strong learner`, provided there are a sufficient number of weak learners and they are sufficiently diverse. This is due to the law of large numbers (think about a biased coin that has a 51% chance of coming up heads). However, this is only true if all classifiers are perfectly independent, making uncorrelated errors. This is hard to achieve, because we train on the same data. A possible solution is to use very different algorithms.

If all classifiers are able to predict class probabilities, we can predict the class with the highest class probability, averaged over all the individual classifiers. This is called `soft voting`. It often reaches higher performance than hard voting because it gives more weight to highly confident votes. 

## Bagging and Pasting

One way to get a diverse set of classifiers (necessary for independency condition) is to train very different algorithms, as described above. Another approach is to use to same training algorithm for every predictor and train them on different random subsets of the training set. This can be done in two ways. `Bagging`, short for bootstrap aggregating, samples with replacement. `Pasting`performs the sampling without replacement. We simply aggregate the predictions of the ensemble by doing a statistical mode in case of classifiers, and an average in case of regressors. Generally, the net result is that the ensemble has a similar bias, but a lower variance than a single predictor trained on the original dataset. An other advantage is that each predictor can be trained in parallel, via different CPU cores, or even different servers. 

Bootstrapping introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting. However, because of this diversity, the predictors end up being less correlated, so the ensemble's variance is reduced. Overall, bagging is generally preferred above pasting. 
