# Logistic Regression and Stochastic Gradient Descent (SGD)

Logistic Regression is one of the most basic Machine Learning algorithms for classification tasks (e.g. spam detection). It models the relationship between a categorical response variable ($Y$) and one or more predictor variables ($X$). Based on the number and data type of the classes, there are different forms of logistic regression:



1. `Binary Logistic Regression`: The target variable has two possible categorical values (e.g. spam vs. no spam).

2. `Multinomial Logistic Regression`: The target variable has three or more possible categorical outcomes (e.g. handwritten digit classifier).

3. `Ordinal Logistic Regression`: The target categorical variables are ordered (e.g. satisfaction ratings).

Logistic Regression forms the perfect introduction to concepts like Stochastic Gradient Descent and classification evaluation metrics.



## Binary Logistic Regression

A binary classifier $h$ is a function that maps from $\mathcal{X}$ (domain of $X \in \mathbb{R}^p$)  to {0, 1}, e.g. $Y = 1$ being "spam" and $Y=0$ being "non-spam" in email classification. The classifier is linear if a linear function $H(X)$ exists such that $h(X) = 1$  if  $H(X) \geq 0$:

$$
H(\textbf{X}) = \beta_0 + \boldsymbol{\beta}^T \textbf{X} \\
\ \\
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
h^*(x) =   \left\{
    \begin{aligned}
      & 1 \quad \text{if} \ \ p(x) > \frac{1}{2}\\
      & 0 \quad \text{otherwise}
    \end{aligned}
  \right.
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
\hat{y} =   \left\{
    \begin{aligned}
      & 1 \quad \text{if} \ \ \hat{p}(\textbf{x})  \geq \frac{1}{2}\\
      & 0 \quad \text{if} \ \ \hat{p}(\textbf{x})  < \frac{1}{2}\\
    \end{aligned}
  \right.
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

This loss function is called the `Log Loss` or `Binary Cross Entropy Loss`. Note that one could also add a regularization term to this loss function, just like in ridge and lasso regression. The next question is, how do we achieve a low value for this loss function?  The answer to this question is an iterative procedure called Stochastic Gradient Descent, which will be explained in the next section.

## Stochastic Gradient Descent

## Multinomial Logistic Regression

Multinomial Logistic Regression extends the logistic regression model to $K > 2$ classes. We again use a logit transformation, where we compare each class to the base class ($K = 0$). This gives:

$$
log\left(\frac{\mathbb{P}(Y = 1 | X = \textbf{x} )}{\mathbb{P}(Y = 0 | X = \textbf{x} )}\right) =  \beta_{1, \ 0} + \boldsymbol{\beta}_1^T \textbf{x} \\
\ \\
\cdots \\
\ \\
log\left(\frac{\mathbb{P}(Y = K-1 | X = \textbf{x} )}{\mathbb{P}(Y = 0 | X = \textbf{x} )}\right) =  \beta_{K-1, \ 0} + \boldsymbol{\beta}_{K-1}^T \textbf{x} \\
$$

or equivalently:

$$
\begin{aligned} p_k (\textbf{x}) = \mathbb{P}(Y = k \ | \ X = \textbf{x} ) = \ & \frac{e^{\beta_{k, \ 0} + \boldsymbol{\beta}_k^T \textbf{x}}}{1 + \sum_l^{K-1} e^{\beta_{l, 0} + \boldsymbol{\beta}_l^T \textbf{x}}}
\quad \quad \text{for} \ \ k = 1, \ \dots,\ K-1 \\
\ \\
= \ & \text{Softmax} \ (0, \ \beta_{1, \ 0} + \boldsymbol{\beta}_1^T \textbf{x}, \ \dots
, \ \beta_{K-1, \ 0} + \boldsymbol{\beta}_{K-1}^T \textbf{x})\end{aligned}
$$

The `Softmax Function` is a multi dimensional version of the sigmoid function and is a fundamental function for classification problems. The loss function in case of multinomial logistic regression is called the `Cross Entropy Loss` and is defined as follows:

$$
\mathcal{L}(\boldsymbol{\beta}) = -\sum_{i = 1}^{n} \ \sum_{k = 0}^{K-1}
 I(y_i = k) \ log(p_k(x_i))
$$

## Model Evaluation
