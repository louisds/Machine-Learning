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

By default, a bagging classifier samples $m$ instances with replacement. This means that only about 63% of all training instances are sampled on average for each predictor. The remaining 37% of the training instances that are not sampled are called `out-of-bag (OOB)`instances. We can evaluate each predictor on these OOB instances, and average them to obtain a general `OOB score` for the ensemble method. 

Instead of sampling from the training set, we could also sample the features. We can define the maximum amount of features to consider (same like the maximum amount of samples) and each predictor will be sampled on a random subset of input features. This technique is particularly useful when dealing with high dimensional inputs. There are two common techniques that use this methodology. `Random Patches Method` samples both training instances and input features. `Random Subspaces Method` only samples from the input features and keeps all training instances.

## Random Forests

`Random forests` are an ensemble of Decesion Trees generally trained via the bagging method. Typically, the max samples is set to the size of the training set. The algorithm introduces extra randomness when growing trees: instead of searching for the very best feature among all features when splitting a node, the algorithm searches for the best feature among a random subset of features. This results in greater tree diversity, which trades a higher bias for a lower variance, generally yielding an overall better model. 

It is possible to make the trees even more random by also using random thresholds for each features rather than searching for the best possible thresholds (like regular decision trees and the CART algorithm do). Such a forest is called `Extra Trees Ensemble`(Extremely Randomized Trees). It trades again a little more bias for a slightly lower variance. An other advantage is that extra trees are faster to train. 

An other aspect of random forests is `feature importance`, which is measured by looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest). More precisely, it is a weighted average, where each node's weight is equal to the number of training samples that are associated with it. 

## Boosting

`Boosting` refers to any ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor. The most popular boosting methods are `AdaBoost`(Adaptive Boosting) and `Gradient Boosting`. 

### AdaBoost

One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. As a result, the new predictors will focus more and more on the hard cases. This is done by changing the relative weight of the missclassified training instances in the loss function. This `sequential learning` technique has some similarities with Gradient Descent, except that instead of tweaking a single predictor's parameters, Adaboost adds predictors to the ensemble, gradually making it better. The downside of this sequential learning technique is that it cannot be parellelized, since each predictor can only be trained after the previous predictor has been trained and evaluated. As a result, it doesn't scale as well as bagging or pasting.

In the AdaBoost algorithm, each instance weight $w^{(i)}$ is initialized to $1/m$, with $m$ the amount of training instances. After the $j^{th}$ predictor is trained, its weighted error rate $r_j$ is computed on the training set as:

$$
r_j = \frac{\sum_{n=1}^{m}w^{(i)}}{\sum^{m} w^{(i)}}}
$$







