# Naive Bayes and Support Vector Machines

In this chapter, we will handle two other classical algorithms that are often used for classification problems: Naive Bayes and Support Vector Machines.  

## Naive Bayes Methods

Naive Bayes methods are a set of algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable (i.e. that the presence of one particular feature does not affect the other). Another basic assumption is that all the predictors have an equal effect on the outcome. Naive Bayes methods are mostly used for classification tasks and belong to the Generative Models. 

Bayes' Theorem states the following relationship:

$$
\mathbb{P}(y | \mathbf{x}) = \frac{\mathbb{P}(y) \ \mathbb{P}(\mathbf{x} | y) }
{\mathbb{P}(\mathbf{x})}
$$

where $\mathbf{x} = (x_1, \ \dots, \ x_p)$ is the $p$-dimensional feature vector. The naive conditional independence assumption is stated as follows:

$$
\mathbb{P}(x_i | y, \ x_1, \ \dots, \ x_{i-1}, \ x_{i+1}, \ \dots, \ x_{p}  )
= \mathbb{P}(x_i | y)
$$

$$
\Rightarrow \mathbb{P}(\mathbf{x} | y) = \prod_{i=1}^p \mathbb{P}(x_i | y)
$$

This simplifies the Bayes' relationship to:

$$
\mathbb{P}(y | \mathbf{x}) = \frac{\mathbb{P}(y) \ \prod_{i=1}^p \mathbb{P}(x_i | y) }
{\mathbb{P}(\mathbf{x})}
$$

Since the denominator $\mathbb{P}(\mathbf{x})$ is just a normalizing constant (also called `evidence`), we can use the following classification rule:

$$
\mathbb{P}(y | \mathbf{x}) \propto \mathbb{P}(y) \ \prod_{i=1}^p \mathbb{P}(x_i | y) 

$$

$$
\Rightarrow \hat{y} = \argmax_y \left( 
\mathbb{P}(y) \ \prod_{i=1}^p \mathbb{P}(x_i | y) 
\right)
$$

The probability $\mathbb{P}(y)$ can be estimated by the relative frequency of each class in the training set. The probability $\mathbb{P}(x_i | y)$ depends on the type of naive Bayes classifier that is chosen, as each type makes different assumptions regarding this distribution. 



#### Gaussian Naive Bayes



`Gaussian Naive Bayes` is one of the most popular naive Bayes methods in case the features are continuous variables. The method assumes that each class follows a Gaussian distribution:

$$
\mathbf{X} = \mathbf{x} \ | \ Y = y_c  \sim N(\boldsymbol{\mu_c}, \ \Sigma_c) \quad 

\quad \text{for} \ c = 1, \ \dots, p
$$

This is similar to Quadratic Discriminant Analysis (QDA), where we are also assuming class-specific covariance matrices. The difference is that Gaussian Naive Bayes assumes independence of the features, which means the covariance matrices are diagonal matrices. This lowers the amount of parameters that have to be estimated. Similarly to LDA and QDA, the mean of each Gaussian is estimated as follows:

$$
\hat{\boldsymbol{\mu_c}} = \frac{1}{N_c} \sum_{i = 1}^n I(y_i = c) \ x_i
$$

Where $N_c$ is the amount of observations belonging to class $c$. In other words, this is just the mean of all datapoints belonging to class $c$. The covariance (diagonal) matrix for each class is estimated as follows:

$$
\hat{\Sigma}_c = \text{diag} \left( \frac{1}{N_c} \sum_{i = 1}^n I(y_i = c) \ 
(\mathbf{x} - \hat{\boldsymbol{\mu_c}})
 (\mathbf{x} - \hat{\boldsymbol{\mu_c}})^T \right)
$$

#### Categorical Naive Bayes

If the feature is a categorical variable, we can use the `Categorical Naive Bayes` method. The method assumes each feature has its own categorical distribution, estimated by:

$$
\mathbb{P}(X = t \ | \ Y = c) = \frac{N_{tc} + \alpha}{N_c + \alpha T} \quad \quad 
\text{where} \ \sum_{t=1}^T N_{tc} = N_c
$$

In words, this is the probability that the categorical feature $X$ belongs to class $t$, given that the target variable $Y$ belongs to class $c$. This probability is estimated by the relative frequency of each category within $X$ for the specific class $c$, which has a total of $N_c$ observations. The absolute frequency of each category is denoted by $N_{tc}$. The smoothing parameter $\alpha > 0$ is added to prevent probabilities equal to zero. Setting $\alpha = 1$ is called `Laplace smoothing`, while setting $\alpha < 1$  is called `Lidstone smoothing`. The number of different categories within $X$ is denoted by $T$. While this example only shows the method for one categorical variable, categorical naive Bayes can be extended to multiple categorical variables. 



#### Conclusion



Some other popular naive Bayes methods include `Multinomial Naive Bayes` and `Bernoulli Naive Bayes`. Both of these methods are popular as a simple text classification method in the domain of `Natural Language Processing`. In case of multinomial naive Bayes, the method assumes multinomially distributed data (for example word vector counts). Bernoulli naive Bayes assumes multivariate Bernoulli distributed data, like word occurrence vectors (rather than word count vectors). Both of these methods will be handled in the Natural Language Processing part of the course. 



To conclude, naive Bayes algorithms are fast and easy to implement. Moreover, they only require a small amount of training data to estimate the parameters. Compared to QDA, Gaussian Naive Bayes is less prone to the curse of dimensionality as the covariance matrix is a diagonal matrix (i.e. less parameters to estimate). Compared to logistic regression, naive Bayes has a higher bias (less complex model), but lower variance.



The biggest disadvantage of naive Bayes methods, however, is the assumption of predictors to be independent. In most of the real life cases, the predictors are dependent. This sometimes hinders the performance of the classifier. 



## Support Vector Machines
