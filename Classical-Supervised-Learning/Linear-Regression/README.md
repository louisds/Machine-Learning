# Linear Regression

## Introduction

Linear regression is a `parametric` statistical modeling technique used to establish the relationship between a continuous dependent variable (`response` variable $y \in \mathbb{R}$) and one or more independent variables (predictors or `covariates` $\textbf{x} \in \mathbb{R}^p$):

$$
y = f(\textbf{x}) + \varepsilon
$$

where $f$ is a linear function of $\textbf{x}$ and $\varepsilon \sim N(0, \ \sigma^2)$ the `error term` or irreducible error, which represents the deviation of the actual value of $y$ from the measured value. The goal of linear regression is to calculate the estimated linear function $\hat{f}$, which is used to derive estimates $\hat{y}$:

$$
\hat{y} = \hat{f}(\textbf{x})
$$

Linear regression is based on a set of assumptions that must be satisfied for the estimates of the coefficients to be valid and reliable. Violations of these assumptions can lead to biased or inefficient estimates, as well as inaccurate predictions and inferences. The four key assumptions are:

1. `Linearity`: Linear relationship between the response and the predictors. The linearity assumption can be checked by examining scatterplots of the dependent variable against each independent variable. If the relationship between the variables is not linear, a nonlinear transformation of the data may be necessary to satisfy the linearity assumption.
2. `Independence`: The errors (and thus the observations) are independent of each other. Violations of the independence assumption can arise in a number of ways. Examples include autocorrelation (the value of the dependent variable at one time point is related to the value at a previous time point), or repeated measures (the same observation is measured multiple times).
3. `Normality`: The errors are normally distributed, such that $\varepsilon \sim N(0, \ \sigma^2)$. One way to check for normality is to examine the distribution of the residuals. If the distribution of the residuals is not approximately normal, this may indicate a violation of the normality assumption.
4. `Homoscedasticity`: The variance of the errors is constant across all levels of the independent variables. Violations of the homoscedasticity assumption can lead to biased estimates of the coefficients and incorrect standard errors. One way to check for homoscedasticity is to examine the residuals (the differences between the observed values of the dependent variable and the predicted values) against the predicted values. If the spread of the residuals is not constant across all predicted values, this may indicate a violation of the homoscedasticity assumption.

Linear regression is starting point in the machine learning journey, as it is the foundation for more sophisticated topics like regularization, support vector machines, and neural networks. 

## Simple Linear Regression

Given the observed data $\mathcal{D} = \set{ (x_1, y_1), \cdots, (x_n, y_n) }$, `simple linear regression` is used to model the relationship between a single dependent variable ($y$) and a single independent variable ($x$):

$$
y = \beta_0 + \beta_1 x + \varepsilon
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

In practice, it is much cleaner and easier to work with vector notations (will be useful later). We denote the `design matrix` $\mathbb{X}$ as:

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
\hat{\boldsymbol{\beta}} = (\mathbb{X}^T \mathbb{X})^{-1} \ \mathbb{X}^T \ \textbf{Y} \\
\hat{\textbf{Y}} = \mathbb{X} \ (\mathbb{X}^T \mathbb{X})^{-1} \ \mathbb{X}^T \ = \mathbb{H} \ \textbf{Y}
$$

where $\mathbb{H}$ is called the `projection matrix` (as $\hat{\textbf{Y}}$ is the projection of $\textbf{Y}$ onto the space spanned by the columns of $\mathbb{X}$). 






