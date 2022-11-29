# Decision Trees

https://climbtheladder.com/random-forest-interview-questions/

https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-random-forests/

https://iq.opengenus.org/questions-on-random-forest/

https://krishnaik.in/2022/03/01/important-interview-questions-on-random-forest-machine-learning-algorithm/

https://medium.com/@penggongting/in-interview-how-to-answer-compare-random-forest-and-gradient-boosting-decision-tree-105de35cff3b

https://www.linkedin.com/pulse/top-interview-question-machine-learning-decision-tree-deepak-chaubey/

https://www.mlstack.cafe/blog/random-forest-interview-questions

## Overview of a Decision Tree

An example of a simple decision tree that classifies flowers from the Iris dataset is shown below.

<img src="https://github.com/louisds/Machine-Learning/blob/main/Classical-Supervised-Learning/Trees-and-Random-Forests/images/tree.png"  width="300">

A decision tree is drawn upside down with its root at the top (root node) and has a certain tree depth (equal to two example). It consists of internal nodes (conditions), based on which the tree splits into edges (branches). The end of the branch that doesn’t split anymore (no child nodes) is called the leaf node (decision). Each node tells us how many $samples$ it contains, with $values$ telling us how much of each class. Finally, a node's Gini attribute measures its impurity. A node is pure (gini=0) if all training instances it applies to belong to the same class. 

## Evaluation Metrics at Nodes

In the example above, which is a classification problem, each node has a Gini impurity score. The Gini score of node $i$ can be calculated as:

$$
G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2
$$

where $p_{i,k}$ is the amount of samples from class $k$ divided by the total amount of training samples in the $i^{th}$ node. Another metric that can be used for classification problems is the entropy score, which is defined as:

$$
H_i = - \sum_{k=1}^{n} p_{i,k} \log{p_{i,k}}
$$

where the $p_{i,k}$-values equal to zero are not accounted in the sum. 



## The Classification and Regression Trees (CART) Algorithm

## Advantages of Decision Trees

* They require very little data preparation. They don't require feauture scaling or centering.
* It can cover both regression and classification.
* 
