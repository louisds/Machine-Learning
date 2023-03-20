# Linear Regression and Regularization

In this chapter, I will explain the most basic and fundamental Machine Learning (ML) algorithm: Linear Regression. This algorithm will be used as a backbone to explain another crucial concept: Regularization.

## Introduction to Linear Regression

Linear regression is a `parametric` statistical modeling technique used to establish the relationship between a continuous dependent variable (`response` variable $Y \in \mathbb{R}$) and one or more independent variables (predictors or `covariates` $\textbf{X} \in \mathbb{R}^p$):

$$
Y = f(\textbf{X}) + \varepsilon
$$

where $f$ is a fixed, unknown function of $\textbf{X}$ and $\varepsilon \sim N(0, \ \sigma^2)$ the `error term` or irreducible error, which represents the deviation (noise) of the actual value of $y$ from the measured value. The best predictor is the regression function:

$$
f(\textbf{x}) = \mathbb{E} \left(Y | \textbf{X} = \textbf{x} \right)
$$

However, the true regression function $f(\textbf{X})$ is not known and hence we need to estimate it. In a linear regression problem, we try to find the `Best Linear Predictor`:

$$
f(\textbf{X}) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
$$

The goal of linear regression is to calculate (or estimate) the coefficients $\beta$, also known as the `weights`, to obtain $\hat{f}(\textbf{X})$ and make predictions $\hat{\textbf{Y}}$. Linear regression is based on a set of assumptions that must be satisfied for the estimates of the coefficients to be valid and reliable. Violations of these assumptions can lead to biased or inefficient estimates, as well as inaccurate predictions and inferences. The four key assumptions are:

1. `Linearity`: Linear relationship between the response and the predictors. The linearity assumption can be checked by examining scatterplots of the dependent variable against each independent variable. If the relationship between the variables is not linear, a nonlinear transformation of the data may be necessary to satisfy the linearity assumption.
2. `Independence`: The errors (and thus the observations) are independent of each other. Violations of the independence assumption can arise in a number of ways. Examples include autocorrelation (the value of the dependent variable at one time point is related to the value at a previous time point), or repeated measures (the same observation is measured multiple times).
3. `Normality`: The errors are normally distributed, such that $\varepsilon \sim N(0, \ \sigma^2)$. One way to check for normality is to examine the distribution of the residuals. If the distribution of the residuals is not approximately normal, this may indicate a violation of the normality assumption.
4. `Homoscedasticity`: The variance of the errors is constant across all levels of the independent variables. Violations of the homoscedasticity assumption can lead to biased estimates of the coefficients and incorrect standard errors. One way to check for homoscedasticity is to examine the residuals (the differences between the observed values of the dependent variable and the predicted values) against the predicted values. If the spread of the residuals is not constant across all predicted values, this may indicate a violation of the homoscedasticity assumption.

Linear regression is starting point in the machine learning journey, as it is the foundation for more sophisticated topics like regularization, support vector machines, and neural networks. 

## Simple Linear Regression

Given the observed data $\mathcal{D} = \set{ (x_1, y_1), \cdots, (x_n, y_n) }$, `Simple Linear Regression` is used to model the relationship between a single dependent variable ($Y$) and a single independent variable ($X$):

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

where $\beta_0$ is the `intercept` (the value of $y$ when $x=0$) and $\beta_1$ the `slope` (the change in $y$ for a one-unit increase in $x$). The goal of simple linear regression is to find estimations $\hat{\beta_0}$ and $\hat{\beta_1}$ as close as possible to the true ones, such that the predicted values $\hat{y}$ are as close as possible to the actual values of $y$. For this purpose, we use a method called least squares regression, where we minimize the sum of the squared residuals (`SSR`):

$$
SSR(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \beta_0 + \beta_1 x_i)^2
$$

where $n$ is the number of observations. We choose $\hat{\beta_0}$ and $\hat{\beta_1}$ such that $SSR(\hat{\beta_0}, \hat{\beta_1})$ is as small as possible. By using some simple calculus, we obtain the following formulas:

$$
\hat{\beta_0} = \bar{y} - \hat{\beta_1} \bar{x}
$$

$$
\hat{\beta_1} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

where $\bar{x}$ and $\bar{y}$ are the mean of respectively x and y.

In practice, it is much cleaner and easier to work with vector notations (will be useful later). We denote the `Design Matrix` $\mathbb{X}$ as:

$$
\mathbb{X} = (\mathbb{1}, X) = \begin{pmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{pmatrix}
$$

Which gives:

$$
\textbf{Y} = \mathbb{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

From a linear algebra point of view, the SSR is defined as:

$$
SSR(\boldsymbol{\beta}) = \lVert \textbf{Y} - \hat{\textbf{Y}} \rVert^2 
$$

The values for \textbf{\hat{\beta}} are calculated as follows:

$$
SSR = (\textbf{Y} - \mathbb{X} \boldsymbol{\beta})^T  \ (\textbf{Y} - \mathbb{X} \boldsymbol{\beta}) = \textbf{Y}^T \textbf{Y} - 2 \boldsymbol{\beta}^T \mathbb{X}^T \textbf{Y} + \boldsymbol{\beta}^T \mathbb{X}^T \mathbb{X} \boldsymbol{\beta}
$$

$$
\Rightarrow \frac{\partial SSR}{\partial \boldsymbol{\beta}} = - 2 \mathbb{X}^T \textbf{Y} + 2 \mathbb{X}^T \mathbb{X} \boldsymbol{\beta} = 0
$$

Which gives:

$$
\hat{\boldsymbol{\beta}} = (\mathbb{X}^T \mathbb{X})^{-1} \ \mathbb{X}^T \ \textbf{Y}
$$

This minimizer $\hat{\boldsymbol{\beta}}$ is called the `Ordinary Least Square (OLS) Estimator` and can be used to make new predictions. We can further write $\hat{\textbf{Y}}$ as follows:

$$
\hat{\textbf{Y}} = \mathbb{X} \ (\mathbb{X}^T \mathbb{X})^{-1} \ \mathbb{X}^T \ \textbf{Y} = \mathbb{H} \ \textbf{Y}
$$

where $\mathbb{H}$ is called the `Projection Matrix` (as $\hat{\textbf{Y}}$ is the projection of $\textbf{Y}$ onto the space spanned by the columns of $\mathbb{X}$). 

## Multiple Linear Regression

In `Multiple Linear Regression`, the feature vector $\textbf{x} \in \mathbb{R}^p$ is multi-dimensional:

$$
y = \beta_0 + \beta_1 \ x_1 +  \beta_2 \ x_2 + \cdots + \beta_p \ x_p + \varepsilon
$$

The design matrix $\mathbb{X} \in \mathbb{R}^{n \ \times \ p+1}$ is given as:

$$
\mathbb{X} = (\mathbb{1}, \textbf{X}) = \begin{pmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p}\\
1 & x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \vdots & \vdots & \ddots  & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}\\
\end{pmatrix}
$$

Due to the similarity of the algebraic expression, the same formula for the least square estimator $\hat{\boldsymbol{\beta}}$ as for simple linear regression can be used.

## Model Validation

We are most interested in how the model will perform on unseen data, but what if we don't have other data. To solve this issue, we can randomly split the data into a training and a test set (for example a 80-20 split). The training data is going to be used to fit the model, while the test set is meant for model evaluation. The most popular evaluation metric in linear regression is the `Mean Square Error (MSE)`, calculated as:

$$
MSE = \frac{SSR}{n} =  \frac{1}{n} \ \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

The smaller the MSE, the closer the predicted values are to the measured observations. Unfortunately, we cannot interpret many insights from one single result but it gives us a real number to compare against other model results and help us select the best regression model. 

While the Mean Square Error (MSE) is an absolute measure of the goodness for the fit, the `R-Squared` value is a relative measure of how well the model fits dependent variables. It measures how much variability in the dependent variable (Y) can be explained by the model. It can be calculated as:

$$
R^2 = 1 - \frac{\sum \left(y_i - \hat{y}_i \right)^2}{\sum \left(y_i - \bar{y}_i \right)^2}
$$

The R-Squared value lies between 0 to 1 (can technically be smaller than 0), with a value closer to 1 indicating a better fit between prediction and actual value. It is a good measure to determine how well the model fits the dependent variables. However, it does not take into consideration of the overfitting problem (see next section). If our regression model has many independent variables (the model is too complicated) it may fit very well to the training data but performs badly for our testing data. That is why `Adjusted R Squared` is introduced because it will penalize additional independent variables added to the model and adjust the metric to prevent overfitting issues:

$$
R^2_{adj} = 1 - \left( \frac{n-1}{n-p-1} \right) \ \frac{\sum \left(y_i - \hat{y}_i \right)^2}{\sum \left(y_i - \bar{y}_i \right)^2}
$$

There are two other (less common) evaluation metrics that are used in practice. The `Root Mean Square Error (RMSE)` is the square root of MSE. In some cases, it is used instead of the MSE for two reasons. Firstly, sometimes the MSE value can be too big to compare easily between models. Secondly, by taking the square root we take the units back to the original units. The `Mean Absolute Error (MAE)` is similar to MSE. However, instead of the sum of square of error in MSE, MAE is taking the sum of the absolute value of error:

$$
MAE = \frac{1}{N} \sum_i^N \left| y_i - \hat{y}_i \right|
$$

MSE gives a larger penalization to big prediction errors (by squaring them), while MAE treats all errors the same.


## Overfitting

One would think that adding more and more predictors will make the model better and better. Unfortunately, this is not the case, because of the danger of overfitting. Roughly speaking, a model is `overfitting` the data when it has a small training MSE but a large test MSE. At this point, the model becomes too complex and starts to fit noise or random fluctuations in the training data rather than the underlying pattern. In other words, the model learns the training data too well and becomes too specific to it, losing its generalization power to unseen data. Overfitting can be a result of adding to many predictors (high model complexity).

Two terms related to overfitting are bias and variance. The `Bias` tells us how much (on average) the model's predictions are off from the true values. A model with **high bias** tends to **underfit** the data, which means that it does not capture the underlying patterns in the data well enough. This is often the result of a model being too simple (i.e. too few parameters). As a result, the model may miss important features or relationships between variables, leading to inaccurate predictions. The bias is calculated as the difference between the estimator's expected value and the true value of the variable being estimated. In most cases, the squared bias is used.

$$
\text{Bias}^2(f, \hat{f}) = \left( \mathbb{E}(\hat{f}) - f \right)^2
$$

Note that the bias of a model can not be calculated exactly, as we don't know the true function (or values). `Variance` of a model refers to the amount by which the predicted values of the model vary (on average) for different training data sets. In other words, it measures how sensitive the model is to changes in the training data. A model with **high variance** tends to **overfit** the data, which means that it captures noise or random fluctuations in the data rather than the underlying pattern. It is calculated as follows:

$$
\text{Var}(\hat{f}) = \mathbb{E}\left[\left(\hat{f} - \mathbb{E}(\hat{f})\right)^2\right]
$$

Bias and variance are two sides of the same coin. As squared bias decreases, the variance goes up (and vice versa). This is called the Bias-Variance Tradeoff and will be explained in the next section. 

## Bias-Variance Tradeoff

Suppose that we obtain some $\hat{f} = \hat{y}$, how well does it estimate $f$? We define the `Prediction Error (or Risk)` using $\hat{f}(\textbf{X})$ as a prediction for $Y$ as:

$$
R(\hat{f}) = \mathbb{E}(Y - \hat{f})^2 = \underbrace{\mathbb{E}(f-\hat{f})^2}_{\text{Reducible Error}} +  \underbrace{\mathbb{E}(Y - f)^2}_{\text{Irreducible Error } \ \sigma^2}
$$

The prediction error, also known as the `Total Error` or the generalization error, has been decomposed into two errors, the reducible and the irreducible error. The `Irreducible Error` (Bayes Error) is essentially noise that we do not want to learn (recall $Var(\varepsilon) = \sigma^2$). The `Reducible Error` is the error that we have some control over. It is often called the mean squared error of estimating $f$ using $\hat{f}$, which can be further decomposed into Squared Bias and Variance:

$$
\text{MSE}(f, \hat{f}) = \mathbb{E}(f-\hat{f})^2 = \mathbb{E}(f - \mathbb{E}\hat{f} + \mathbb{E}\hat{f} - \hat{f})^2 = \mathbb{E}(f - \mathbb{E}\hat{f})^2 + \mathbb{E}(\hat{f} - \mathbb{E}\hat{f})^2 - 2  \mathbb{E}(f - \mathbb{E}\hat{f})(\hat{f} - \mathbb{E}\hat{f}) 
$$

$$
\Rightarrow \text{MSE}(f, \hat{f}) = \text{Bias}^2(f, \hat{f}) + \text{Var}(\hat{f})
$$

This expression is known as the `Bias-Variance Decomposition`. In a perfect world, we would be able to find some $\hat{f}$ that is unbiased (i.e. bias equals zero) and also has a low variance. However, in practice, this isn’t always possible. It turns out that there is a `Bias-Variance Tradeoff`. That is, often, the less bias in our estimation, the higher the variance (and vice versa). Complex models tend to be unbiased, but highly variable. Simple models are often extremely biased, but have low variance. This relation is shown in the sketch below.

<p align="center"> <img src="https://github.com/louisds/Machine-Learning/blob/main/Classical-Supervised-Learning/images/BV-Tradeoff.png"  width="500"> </p>

To select a model that appropriately balances the tradeoff between bias and variance, and thus minimizes the reducible error, we need to select a model of the appropriate complexity for the data. OLS estimators are unbiased estimators which lead to the lowest SSR, but is a model with the lowest SSR always the best model? The answer to this question is no, sometimes we need to introduce some Bias to make the model more general and prevent overfitting. In the sketch, this is shown by the OLS estimator being more rightly located than the sweet spot (i.e. lowest total error). To solve this issue, we want to "add bias" to lower the variance and come closer to the point of minimal risk. This concept is called `Regularization`. Regularization adds a penalty to the loss function to prevent the weights $\beta$ from becoming too large (i.e. making them smaller by penalizing large weights). In this way, the variance will decrease, which means the model will be more generalizible and less capable of fitting random noise. There are two popular regularization methods in Linear Regression, Ridge Regression and Lasso Regression. Both concepts will be explained in the next section. 

## Ridge Regression (L2 Regularization)

Recall that the OLS estimator minimizes the SSR or training error (i.e. loss function). In case of `Ridge Regression`, the training error is penalized by adding an extra term:

$$
\hat{\boldsymbol{\beta}} = \text{Argmin_{\boldsymbol{\beta}}} \left( \lVert \textbf{Y} - \hat{\textbf{Y}} \rVert^2 \right)
$$















