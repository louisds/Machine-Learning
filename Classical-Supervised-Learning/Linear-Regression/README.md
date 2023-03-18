# Linear Regression

## Introduction

Linear regression is a parametric statistical modeling technique used to establish the relationship between a continuous dependent variable (response variable $y \in \mathbb{R}$) and one or more independent variables (Predictors $**x** \in \mathbb{R}^p$):

$$
y = f(**x**) + \varepsilon
$$

where f is an unknown function and $\epsilon$ the error term or irreducible error, which represents the deviation of the actual value of $y$ from the measured value. Linear regression is starting point in the machine learning journey, as it is the foundation for more sophisticated topics like regularization, support vector machines, and neural networks. 

Linear regression is based on a set of assumptions that must be satisfied for the estimates of the coefficients to be valid and reliable. Violations of these assumptions can lead to biased or inefficient estimates, as well as inaccurate predictions and inferences. The four key assumptions of linear regression are:

1. **Linearity:** Linear relationship between the response and the predictors
2. **Independence:** The errors (and thus the observations) are independent of each other
3. **Normality:** The errors are normally distributed, such that $\varepsilon \sim N(0, \: \sigma^2)$
4. **Homoscedasticity:** The variance of the errors is constant across all levels of the independent variables

## Simple Linear Regression



