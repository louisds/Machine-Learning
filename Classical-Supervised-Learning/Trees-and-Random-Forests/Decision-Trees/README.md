# Decision Trees

## Overview of a Decision Tree

An example of a simple decision tree that classifies flowers from the Iris dataset is shown below.

<img src="https://github.com/louisds/Machine-Learning/blob/main/Classical-Supervised-Learning/Trees-and-Random-Forests/images/tree.png"  width="300">

A decision tree is drawn upside down with its root at the top (root node) and has a certain tree depth (equal to two example). It consists of internal nodes (conditions), based on which the tree splits into edges (branches). The end of the branch that doesn’t split anymore (no child nodes) is called the leaf node (decision). Each node tells us how many $samples$ it contains, with $values$ telling us how much of each class. Finally, a node's Gini impurity attribute measures its impurity. A node is pure (gini impurity=0) if all training instances it applies to belong to the same class. A decision tree is a non-parametric model because the number of parameters is not determined prior to training. 

Decision trees can also be used for regression problems. Instead of choosing the most occurring class in a leaf node as the prediction, the mean is taken of the instances associated with this leaf node. 

## Evaluation Metrics at Nodes

In the example above, which is a classification problem, each node has a Gini impurity score. The Gini impurity score of node $i$ can be calculated as:

$$
G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2
$$

where $p_{i,k}$ is the amount of samples from class $k$ divided by the total amount of training samples in the $i^{th}$ node. Another metric that can be used for classification problems is the entropy score (1 - information gain), which is defined as:

$$
H_i = - \sum_{k=1}^{n} p_{i,k} \log{(p_{i,k})}
$$

where the $p_{i,k}$-values equal to zero are not accounted in the sum. Most of the time, both metrics lead to similar trees and they do not make a big difference (Gini impurity score is a little faster to calculate so this would be a good default). When they differ, Gine score tends to isolate the most frequent class in its own branch of the tree (like the example above), while entropy tends to produce more balanced trees. For regression problems, a popular choice is the MSE-loss. 

## The Classification and Regression Trees (CART) Algorithm

Scikit-Learn uses the CART algorithm to create (aka. grow) its trees. The algorithm works by first splitting the training set into two subsets using a single feature $k$ and a threshold $t_k$. It searches the pair $(k, t_k)$ that produces the purest subsets (weighted by their size). The CART cost function for classification when using the Gini impurity score is given by:

$$
J(k, t_k) = \frac{m_{left}}{m} G_{left} + \frac{m_{right}}{m} G_{right}
$$

where $m$ is the total amount of instances in the current level. Once the CART algorithm has successfully split the training set in two, it splits the subset using the same logic, and so on, recursively. It stops once it reaches the maximum depth, or if it cannot find a split that reduces the defined node metric (e.g. Gini impurity in this case). The CART algorithm is a greedy algorithm, as it does not check if the split will lead to the lowest possible impurity several levels down, hence, it will not always be the optimal solution. The latter is a NP-complete problem that requires O(exp(m)) time. The CART-algorithm only produces binary trees (nonlead nodes always have two children). Other algorithms, such as ID3, can produces trees where nodes can have more than two children.

Other algorithms work by first training the DT without restrictions, and then pruning unnecessary nodes. A node is unnecessary if the impurity improvement is not statistically significant. This can, for example, be done by a Chi-square test to estimate the probability that the impurity improvement is purely due to random chance (p-value). 

## Computational Complexity

Making a prediction requires us to go from the root node to a leaf. As decision trees are generally balanced, traversing the tree requires going to roughly O($log_2(m)$) nodes. As each node only checks one feature, the overall prediction complexity is O($log_2(m)$), independent of the number of features. This means the predictions are very fast.

The training process, however, has to compare all features on all samples at each node, since it has to decide where to make the split. This leads to a training complexity of $O(n \times m \log_2(m))$. For small training sets (e.g. less than a few thousand instances), Skicit-Learn can speed up the training by presorting the data, but doing that slows down the training considerably for larger training sets. 

## Regularization Hyperparameters

As decision trees are non-parametric models, they are prone to overfitting (i.e. adapting itself to training data if left unconstrained). To avoid overfitting, we need to restrict the DT's freedom during training. The most straightforward way of doing is, is by defining a max depth of the tree. Other ways are:

* Defining the minimum number of samples a node must have before it can split.
* Defining the minimum number of samples a leaf node must have.
* Defining the maximum number of leaf nodes.
* Defining the maximum number of features that are evaluated for splitting at each node.

## Advantages of Decision Trees

* They require very little data preparation. They don't require feauture scaling or centering.
* It can cover both regression and classification.
* Easy interpretation.

## Limitations of Decision Trees

* They can only make splits perpendicular to an axis, which makes them vulnerable to training set rotation (i.e. they can't make diagonal splits). One way to limit this problem is by using PCA, which often results in a better orientation of the training data. 
* Decision trees are also very sensitive to small variations in the training data. Removing a single observation can already drastically change the decision tree, hence leading to a high variance. This instability can be reduced by averaging predictions over many trees (random forests). 
* Decision trees can create biased trees if some classes dominate.
* The CART-algorithm is a greedy algorithm.

## Interview Questions



