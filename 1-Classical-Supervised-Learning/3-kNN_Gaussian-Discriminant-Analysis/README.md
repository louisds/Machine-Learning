# k-Nearest Neighbors and Gaussian Discriminant Analysis

## K-Nearest Neighbors (k-NN)

#### Parametric vs. Non-Parametric Models

Linear and Logistic regression both belong to the category of `Parametric Models`, i.e. models that use a fixed set of parameters. We typically make an assumpting with regards to the form of a function $f$, for example a linear model in case of linear regression, and estimate the set of parameters. 

The biggest disadvantage of parametric models is that the assumptions we make may not always be true. Hence, these models are usually used for less complex problems, as their algorithms are less flexible. On the positive side, parametric models are more interpretable and require significantly less data than non-parametric models.

`Non-Parametric Models` do not make any underlying assumptions with respect to the form of a function. Instead, they directly use the data to estimate a function $m$ that could be of any form. Parametric models are more flexible, but come with a couple of drawbacks. First of all, they require a lot of training data to accurately estimate the function $m$. Secondly, they are very prone to overfitting. The most basic example of a non-parametric model is the k-Nearest Neighbors (k-NN) algorithm, which will be explained in this chapter. 

#### k-NN Algorithm

The `k-Nearest Neighbors (k-NN)` algorithm is a non-parametric model that can be used for both regression and classification. The model is based on the on the assumption that similar observations are more likely to belong to the same class or have similar values. The algorithm itself is pretty simple: the classification or regression of a new instance is determined by the most common class or the mean value of its k nearest neighbors in the feature space. Note that a small value of $k$ could lead to overfitting, while a big value of $k$ can lead to underfitting. When calculating the distance between points, it is recommended to standardize the variables to prevent features with higher orders of magnitude dominating the distance.

For the algorithm to work best on a particular dataset, we need to choose the most appropriate way of calculating the distance between two observations. The most popular distance metric is called the `Euclidian Distance`, which is a measure of the true straight line distance between two points in a $p$-dimensional Euclidean space:

$$
d_{Euclidian} (a, b) = \sqrt{\sum_{i=1}^{p} (a_i - b_i)^2}
$$

In case of high dimensional data (high $p$), or when you want to place less emphasis on outliers, the `Manhattan Distance`, also known as the city block distance, is preferred. It measures the distance of the grid-like path from point $a$ to point $b$: 

$$
d_{Manhattan} (a, b) = \sum_{i=1}^{p} | a_i - b_i |
$$

A more general distance metric is the `Minkowski Distance`:

$$
d_{Minkowski} (a, b) = \left(\sum_{i=1}^{p} | a_i - b_i |^q\right)^{-1/q}
$$

which depends on the choice of the order $q$ of the distance metric. For $q=1$ and $q=2$, this distance metric reduces to respectively the Manhattan and Euclidian distance. High values of $q$ will lead to more emphasis on the larger differences between the corresponding elements of two vectors, with the most extreme version being the `Chebyshev distance`, where $q \rightarrow \infty$:

$$
d_{Chebyshev} (a, b) =  \lim_{q \rightarrow \infty} \left(\sum_{i=1}^{p} | a_i - b_i |^q\right)^{-1/q}
= \max_i \left(|a_i - b_i| \right)
$$

Another distance that is popular in case of textual data analysis (e.g. document or word similarity) is the `Cosine Distance` or Cosine Similarity, which  measures the cosine of the angle between two vectors:

$$
d_{Cosine} (a, b) = \cos{(\theta)} = \frac{\sum a_i \ b_i}{\sqrt{\sum a_i^2}
\ \sqrt{\sum b_i^2}}
$$

So far, we have only covered the distance metrics that are used when we are dealing with continuous variables. But what if we have categorical variables? This is where we can make use of another distance metric called Hamming Distance. The `Hamming Distance`  represents number of positions at which the corresponding elements of two vectors are different. So the larger the Hamming Distance between two vectors, the more dissimilar they are. 

To conclude, the k-NN algorithm is a simple and easy ML model to interpret and since it does not make any assumption, it can be used to solve non-linear tasks as well. On the other side, the algorithm becomes very slow as the number of training data points increases because the model needs to store all data points and calculate the individual distances. The algorithm is also sensitive to outliers. 

## Gaussian Discriminant Analysis

#### Discriminative vs. Generative Classifiers

Generally speaking, classification models can be subdivived into two categories: discriminative and generative classifiers. `Discriminative Models`, like logistic regression and k-NN, only model the output $Y$ given a certain $X$. In other words, they model the decision boundary between the classes by calculating the model parameters that maximimze the conditional probability $\mathbb{P}(Y | X = x)$. 

While discriminative models directly learn the conditional probability $\mathbb{P}(Y | X = x)$, `Generative Models` take an extra step and calculate the joint probability (or likelihood) $\mathbb{P}(X , \ Y) = \mathbb{P}(Y) \ \mathbb{P}(X|Y) $. Afterwards, they use Bayes theorem to go to the conditional probability and make a prediction. In other words, generative classifiers model both the input $X$ and the output $Y$. They focus on how input and output occur together and are able to explain how the data is generated. In this way, generative models are able to generate new data instances. 

Although the name is probably confusing, one of the most basic example of a generative model is `Gaussian Discriminant Analysis` (GDA). The big assumption of GDA is that the observations of each class are samples from a multidimensional normal distribution.

GDA comes in two flavors: Linear and Quadratic. Both types will be explained in the next sections. Note that a binary classification problem will be used as an example, but the algorithms are both extendable to multinomial classification problems. 

#### Linear Discriminant Analysis (LDA)

In case of a binary `Linear Discriminant Analysis (LDA)` we assume the following distributions for $\mathbb{P}(Y)$, i.e. the prior distribution, and $\mathbb{P}(X|Y=y)$ :

$$
Y \sim \text{Bernoulli}(\pi)
$$

$$
X | \ y = 1 \sim N(\mu_1, \Sigma)
$$

$$
X | \ y = 0 \sim N(\mu_0, \Sigma)
$$

where the covariance matrix $\Sigma$ of each normal distribution is the same. In words: LDA assumes that each point $\mathbf{x}$ is generated by first doing a Bernoulli trail to obtain $y$ and then sampling from the $p$-dimensional normal distribution corresponding to the class of $y$. If we would repeat this many times, we will get a ton of datapoints in the $p$-dimensional space. The distribution of this data, provided we have enough of it, will be typical of the specific model that we are generating from.

LDA reverses this logic and tries to estimate the parameters of the normal distributions based on the data we have available. After estimating the parameters, which is a simple statistical estimation problem, we can use Bayes' theorem to predict the class $\hat{y}$ that a certain datapoint $\mathbf{x}$ belongs to:

$$
\mathbb{P}(Y = y_i \ | \ X = \mathbf{x}) = \frac{\mathbb{P}(\mathbf{x} \ | \ y_i) 
\ \mathbb{P}(y_i)}
{\mathbb{P}(\mathbf{x})} \propto \mathbb{P}(\mathbf{x} \ | \ y_i) 
\ \mathbb{P}(y_i)
$$

$$
\begin{aligned}\hat{y} = & \ \argmax_i \left(\mathbb{P}(\mathbf{x} \ | \ y_i) 
\ \mathbb{P}(y_i) \right) \\
= & \ \argmax_i \left(log(\mathbb{P}(\mathbf{x} \ | \ y_i) )
+ log(\mathbb{P}(y_i)) \right) \\
= & \ \argmax_i \left(\mathbf{x}^T \hat{\Sigma}^{-1} \hat{\mu}_i - 
\frac{1}{2} \hat{\mu}_i ^T \hat{\Sigma}^{-1} \hat{\mu}_i
+ log(\mathbb{P}(y_i)) \right) \\
\end{aligned}
$$

where the natural logarithm is used to make the problem simpler (log is a monotone increasing function so it will have the same argmax result). The resulting function is also called the discriminant function and is denoted as follows:

$$
\delta_i(\mathbf{x}) = \mathbf{x}^T \hat{\Sigma}^{-1} \hat{\mu}_i - 
\frac{1}{2} \hat{\mu}_i ^T \hat{\Sigma}^{-1} \hat{\mu}_i
+ log(\mathbb{P}(y_i))
$$

So in other words, we pick the class with the highest discriminant function. To obtain the decision boundary, we determine the set of points where the discriminant functions are equal. In case of LDA the decision boundary itself will be linear. Note that after fitting the model, we can use the normal distributions themselves to generate new data.

To conclude, LDA is a quick and easy generative model that can both classify datapoints and generate new ones. Linear boundaries, however, are not always the right solution and sometimes the problem needs a more complex decision boundary. 

#### Quadratic Discriminant Analysis (QDA)

In case of `Quadratic Discriminant Analysis (QDQ)` we do not assume that the covariance matrices are the same:

$$
Y \sim \text{Bernoulli}(\pi)
$$

$$
X | \ y = 1 \sim N(\mu_1, \Sigma_1)
$$

$$
X | \ y = 0 \sim N(\mu_0, \Sigma_0)
$$

As a result, by doing the same calculation as above, the resulting discriminant function is a quadratic function of $\mathbf{x}$:

$$
\delta_i(\mathbf{x}) = - \frac{1}{2} \log | \hat{\Sigma}^{-1} |  - 
\frac{1}{2} (\mathbf{x} - \hat{\mu}_i) ^T \hat{\Sigma}^{-1} 
(\mathbf{x} - \hat{\mu}_i)
+ log(\mathbb{P}(y_i))
$$

In this case, the decision boundary will be a quadratic surface. However, while QDA accommodates more flexible decision boundaries compared to LDA, the number of parameters that need to be estimated is much higher.  The number of parameters that have to be estimated in LDA increases linearly with $p$ while that of QDA increases quadratically with $p$. 
