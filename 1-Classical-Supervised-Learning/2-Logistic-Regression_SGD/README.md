# Logistic Regression and Stochastic Gradient Descent (SGD)

Logistic Regression is one of the most basic Machine Learning algorithms for classification tasks (e.g. spam detection). It models the relationship between a categorical response variable ($Y$) and one or more predictor variables ($X$). Based on the number and data type of the classes, there are different forms of logistic regression:

1. `Binary Logistic Regression`: The target variable has two possible categorical values (e.g. spam vs. no spam).

2. `Multinomial Logistic Regression`: The target variable has three or more possible categorical outcomes (e.g. handwritten digit classifier).

3. `Ordinal Logistic Regression`: The target categorical variables are ordered (e.g. satisfaction ratings).

Logistic Regression forms the perfect introduction to concepts like Stochastic Gradient Descent and classification evaluation metrics.

## Binary Logistic Regression

A binary classifier $h$ is a function that maps from $\mathcal{X}$ (domain of $X \in \mathbb{R}^p$)  to {0, 1}, e.g. $Y = 1$ being "spam" and $Y=0$ being "non-spam" in email classification. The classifier is linear if a linear function $H(X)$ exists such that $h(X) = 1$  if  $H(X) \geq 0$:

$$
H(\textbf{X}) = \beta_0 + \boldsymbol{\beta}^T \textbf{X} 
$$

$$
h(\textbf{X}) = I(H(\textbf{X}) \geq 0)
$$

where $I(\cdot)$ is the indicator function. The linear function $H(X)$ is also known as the `Linear Discriminant Function` , it defines a decision boundary (hyperplane) that separates the two classes. Every classifier comes with a `Classification Risk R` or error rate, that is defined as:

$$
R(h) = \mathbb{P} (Y \neq h(\textbf{X}))
$$

or the probability that the outcome of the classifier is not equal to the true label $Y$. As we only have a specific dataset, we can't calculate this risk exactly and we have to estimate it by the `Empirical Classification Error`  or training error:

$$
\hat{R}(h) = \frac{1}{n} \sum_{i = 1}^{n} I (h(x_i) \neq y_i)
$$

which is basically just counting the amount of mistakes that our model makes and taking an average of it. The optimal classification rule that minimizes $R$ is called `Bayes Rule` and is defined as:

$$
h^*(x) =   \begin{cases}
    \begin{aligned}
      & 1 \quad \text{if} \ \ p(x) > \frac{1}{2}\\
      & 0 \quad \text{otherwise}
    \end{aligned}
  \end{cases}
$$

where $p(X)$ denotes the regression function, which is the probablity that the outcome of $X$ belongs to the default class (we choose $Y = 1$ to be our default class):

$$
p(x) = \mathbb{E}(Y | X = x) = \mathbb{P}(Y = 1 | X = x)
$$

How do we model $p(x)$ ?  Especially because $p(x)$ maps $x$ into the domain $[0, 1]$. For this purpose we perform a `Logit Transformation` :

$$
logit(p) = log\left(\frac{p}{1-p}\right) = log\left(\frac{\mathbb{P}(Y = 1 | X = \textbf{x} )}{\mathbb{P}(Y = 0 | X = \textbf{x} )}\right)
$$

This logit transformation (log-odds) is monotone and maps the interval $[0, 1]$ to $(-\infty, +\infty)$, which is easier for a regression function to map to. In case of logistic regression, this will be modeled as a linear regression model:

$$
H(\textbf{x}) = logit(p(\textbf{x})) = \beta_0 + \boldsymbol{\beta}^T \textbf{x}
$$

where $H(\textbf{x})$ is the Linear Discriminant Function which we talked about before. This shows that the logistic regression boundary is linear in $\mathbf{x}$, as $Y=1$ if  $\hat{H}(\textbf{x}) \geq 0$ and vice versa. Equivalently, the log-odds can be mapped back to a probability:

$$
p(\textbf{x}) = \frac{e^{\beta_0 + \boldsymbol{\beta}^T \textbf{x}}}{1 + e^{\beta_0 + \boldsymbol{\beta}^T \textbf{x}}} = 
\frac{1}{1 + e^{-H(\mathbb{x})}}
 = sigmoid(H(\mathbb{x}))
$$

which is the logistic regression function, also called the `sigmoid function` (or soft-max in case of multinomial classification). The sigmoid function takes any real value as an argument and maps it to a range between 0 and 1. After estimating the weights $\boldsymbol{\hat{\beta}}$, the following decision rule can be made:

$$
\hat{y} =   \begin{cases}
    \begin{aligned}
      & 1 \quad \text{if} \ \ \hat{p}(\textbf{x})  \geq \frac{1}{2}\\
      & 0 \quad \text{if} \ \ \hat{p}(\textbf{x})  < \frac{1}{2}\\
    \end{aligned}
  \end{cases}
$$

## Fitting a Logistic Regression Model

Now that we defined the logistic regression model, we still need a way to estimate the weights $\beta$. Unfortunately, there is no closed form solution. Instead, we use the `Maximum Likelihood Estimation`. In case of MLE, we start with the likelihood of a single observation $(x_i, y_i)$:

$$
L_i(\boldsymbol{\beta}) = p(x_i)^{y_i} \ \left(1-p(x_i) \ \right)^{1-y_i}
$$

We then take the log to obtain the log-likelihood:

$$
\mathcal{l}_i(\boldsymbol{\beta}) = y_i \ log(p(x_i))\ + \ (1-y_i) \ log\left(1-p(x_i) \ \right)
$$

Finally, we aggregate the log-likelihood over all the observations and take the negative (maximizing a function is minimizing the negative of that function) to obtain the loss function:

$$
\mathcal{L}(\boldsymbol{\beta}) = -\sum y_i \ log(p(x_i))\ + \ (1-y_i) \ log\left(1-p(x_i) \ \right)
$$

This loss function is called the `Log Loss` or `Binary Cross Entropy Loss`. Note that one could also add a regularization term to this loss function, just like in ridge and lasso regression. The next question is, how do we achieve a low value for this loss function? The standard (or most basic) approach is Newton's algorithm, where repetitivly a weighted least square regression is performed on a quadratic approximation. Newton's algorithm, however, will fail when we want to fit a big model (large $n$ and $p$). A more scalable iterative approach is gradient descent, which will be explained in the next section.

## Gradient Descent

Gradient Descent is a first-order, iterative, optimization algorithm to find the minimum of a given function, which, in our case, is the loss function $\mathcal{L}(\boldsymbol{\beta})$. The loss function (or loss landscape) can be seen as a bowl, and the goal is to reach the minimum.  To achieve this goal, repeat the following steps iteratively for all the weights $\beta$:

1. Compute the slope (gradient) of the loss function, i.e. first-order derivative, at the current point.

2. For a positive gradient, the loss goes down by decreasing $\beta$ (and vice versa), so update the weights in the opposite direction of the gradient.

One full iteration over the whole training set is called an `Epoch`. Mathematically, the update procedure can be written as follows:

$$
\beta_j \leftarrow \beta_j - \eta \ \frac{\partial \mathcal{L}(\boldsymbol{\mathbf{x}, y,\beta})}
{\partial \beta_j}
$$

where $\eta$ is the `Learning Rate` or step size. It is a key parameter is gradient descent that decides how much we update the parameters (i.e. how far are we going down the slope). This parameter needs to be chosen carefully: if the learning rate is too small, the training will take a very long time to converge. On the other hand, if the learning rate is too high, there is a chance we jump over the minimum and convergence will never be reached. A good practice is to lower the learning rate every few epochs. 

To calculate the gradient $\partial \mathcal{L} / \partial \beta_j$, there are different approaches:

- `Batch Gradient Descent`: The entire training set is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters.

- `Stochastic Gradient Descent`: Instead of using the whole training set, we consider one observation at a time and use that gradient to update the weights.

- `Mini-Batch Gradient Descent`: Use a batch of a fixed number of training examples (mini-batch) to approximate the gradient of Batch Gradient Descent. This option is sometimes refered to as Stochastic Gradient Descent as well.

Each method has its own advantages and disadvantages. BGD often converges (especially in case of convex or relatively smooth error manifolds) to the global minimum of the loss function, whereas SGD has a higher chance of converging to a local minimum. This is because the gradient estimate in SGD is noisier and has more variance, which causes the optimization to jump around in the loss function space. The drawback of BGD is that is slow and computationally expensive (requires a lot of memory), which means it is not suggested for huge training samples. 

By using Mini-Batch Gradient Descent, we can play around with the size of the mini-batch and find a balance between BGD and SGD. A lower `Batch Size` will lead to rapid learning, but a more volatile learning process with higher variance and a chance of converging to a local minimum. 

## Multinomial Logistic Regression

Multinomial Logistic Regression extends the logistic regression model to $K > 2$ classes. We again use a logit transformation, where we compare each class to the base class ($K = 0$). This gives:

$$
log\left(\frac{\mathbb{P}(Y = 1 | X = \textbf{x} )}{\mathbb{P}(Y = 0 | X = \textbf{x} )}\right) =  \beta_{1, \ 0} + \boldsymbol{\beta}_1^T \textbf{x} \\
$$

$$
...
$$

$$
log\left(\frac{\mathbb{P}(Y = K-1 | X = \textbf{x} )}{\mathbb{P}(Y = 0 | X = \textbf{x} )}\right) =  \beta_{K-1, \ 0} + \boldsymbol{\beta}_{K-1}^T \textbf{x} \\
$$

or equivalently:

$$
\begin{aligned} p_k (\textbf{x}) = \mathbb{P}(Y = k \ | \ X = \textbf{x} ) = 
& \begin{cases} \frac{1}{1 + \sum e^{\beta_{l, 0} + \boldsymbol{\beta}_l^T \textbf{x}}}
\quad \quad \text{for} \ \ k = 0 \\ \\ 
\ \\
\frac{e^{\beta_{k, \ 0} + \boldsymbol{\beta}_k^T \textbf{x}}}{1 + \sum e^{\beta_{l, 0} + \boldsymbol{\beta}_l^T \textbf{x}}}
\quad \quad \text{for} \ \ k = 1, \ \dots,\ K-1 \\ \end{cases}
\ \\
\ \\
= \ & \text{Softmax} \ (0, \ \beta_{1, \ 0} + \boldsymbol{\beta}_1^T \textbf{x}, \ \dots
, \ \beta_{K-1, \ 0} + \boldsymbol{\beta}_{K-1}^T \textbf{x})\end{aligned}
$$

The `Softmax Function` is a multi dimensional version of the sigmoid function and is a fundamental function for classification problems. We choose the class with the highest probability as the predicted class. The loss function in case of multinomial logistic regression is called the `Cross Entropy Loss` and is defined as follows:

$$
\mathcal{L}(\boldsymbol{\beta}) = -\sum_{i = 1}^{n} \ \sum_{k = 0}^{K-1}
 I(y_i = k) \ log(p_k(x_i))
$$

## Model Evaluation

How do we evaluate the performance of a classification model? The most straightforward metric is the `Accuracy`, which gives the percentage of correctly classified samples:

$$
\text{Accuracy} = \frac{1}{n} \sum_{i = 1}^{n} I (\hat{y}_i = y_i)
$$

One should be aware that only using accuracy as the metric for model evaluation is a bad habit. For example, in case of an unbalanced dataset (e.g. 90% spam emails and 10% non-spam), the classifier will have a high accuracy just by always guessing "spam". The model itself, however, does not perform and generalize well. A solution is to take a `Weighted Accuracy` where an average is taken of the accuracies for each individual class. In practice it is recommended to also include other classification metrics in the result (or while training). The most common metrics for a binary classification problem will be explained below.

#### Precision

`Precision` is the accuracy of positive predictions. Of all positive predictions (e.g. mails flagged as spam), how many were correct. 

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

* TP = True Positive = Spam emails flagged as spam

* FP = False Positive = Non-spam emails flagged as spam

#### Recall (aka. sensitivity, hit rate, TPR)

`Recall` is the ratio of positive instances that are correctly classified by the classifier. Of spam emails, how many are correctly flagged as spam.

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

* FN = False Negative = Spam emails flagged as non-spam

#### F1-score

`F1-score` is a metric that combines Precision and Recall by taking the harmonic mean (gives more weight to low values than regular mean). 

$$
F_1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}
$$

The classifier will only get a high $F_1$-score if both recall and precision are high.

#### Precision/Recall Trade-Off

The $F_1$-score favors classifiers that have similar precision and recall, but this is not always wanted. Two examples:

* If we would train a classifier to flag videos as being safe for kids, we want to reduce the amount of non-safe videos being flagged as safe (FP), so we want a high precision. The trade-off is that we will have a lot of safe videos being falsely rejected (low recall).

* If we would train a classifier to detect thiefs in shops, we want to reduce the amount of thiefs that are flagged as non-thiefs (FN), so we want a high recall. The trade-off is that we will have a lot of non-thiefs being falsely flagged as thiefs (low precision).

Moral of the story: increasing precision reduces recall, and vice versa. If we increase the `Decision Threshold` (e.g. higher than 0.5), we are being stricter and we will have more FNs (e.g. flagging safe videos as non-safe), which lowers the recall. However, it lowers the FPs (e.g. flagging non-safe videos as safe), hence a higher precision.

Overall, increasing the threshold will lower the recall (always the case) and increase the precision (sometimes it can go down a bit as well, as it depends on total of predicted TRUE-values).

The `PR (Precision-Recall) curve` plots the precision against recall. It is constructed by calculating the precision and recall for different values of the threshold and is a (most of the time) decreasing function - as precision decreases for decreasing recall.

#### Specificity (aka. selectivity, TNR)

The `Specificity` or `Selectivity` is the ratio of negative instances that are correctly classified as negative. Of all negative predictions (e.g. mails flagged as non-spam), how many were correct. In other words, what is the percentage of non-spam mails that were flagged as non-spam. 

$$
\text{Specificity} = \frac{TN}{TN+FP}
$$

* TN = True Negative = Non-spam emails flagged as non-spam

#### False alarm ratio (aka. fall-out, FPR)

The `False Alarm Ratio` is the ratio of negative instances that are incorrectly classified as positive. Of all negative predictions (e.g. mails flagged as non-spam), how many were incorrect. In other words, what is the percentage of non-spam mails that were flagged as spam. 

$$
\text{False Alarm Ratio} = \frac{FP}{TN+FP} = 1 - \text{Specificity}
$$

#### Confusion Matrix

 A confusion matrix is a $(k \times k)$ table - with $k$ the amount of different classes - that serves as a great visual tool to show how well your model performed and which mistakes it made. The table has the actual class labels as the rows and the predicted class labels as the columns.

#### ROC-curve and AUC

The `ROC (Receiver Operating Characteristic) Curve` plots the TPR (Recall) against the FPR (False alarm ratio). This is an increasing function, as the higher the recall (TPR), the more false positives (FPs) the classifier produces. The diagonal of the diagram is the ROC curve of a purely random classifier. A good classifier stays away as far as possible from this line (towards the top left corner).

The `AUC (Area Under the Curve)` is a way of comparing the ROC curve of two classifiers. A perfect classifier will for example have an AUC equal to one. A random classifier will have an AUC equal to 0.5. 

#### PR vs. ROC curve

Choose the PR-curve whenever the positive class is rare (e.g. not a lot of spam mails) or when you care more about FPs (e.g. flagging non-spam mails as spam) than FNs (e.g. flagging spam mails as non-spam). The ROC curve might look really good (e.g. high AUC), but this can be due to the fact that there are only a few observations of the positive class. 

## Conclusion

Logistic Regression is one of the most basic classification algorithms and can be used for both binary and multiclass classification. As there is no closed form solution, an iterative procedure, called Stochastic Gradient Descent, has to be used. 
