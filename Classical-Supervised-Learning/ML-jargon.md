# Bias-Variance

## Bias

The difference between the estimator's expected value and the true value of the variable being estimated.

$$\text{Bias}(\hat{f}) = \mathbb{E}(\hat{f}) - f$$

## Variance

$$\text{Var}(\hat{f}) = \mathbb{E}\left[\left(\hat{f} - \mathbb{E}(\hat{f})\right)^2\right]$$

## Bias-Variance Decomposition

The Bias variance decomposition can be written as:

$$\mathbb{E}(f-\hat{f})^2 = Bias(\hat{f})^2 + Var(f)$$

The response variable `y` can be defined as a true function `f` plus noise $\varepsilon$. This can be written as follows:

$$y = f + \varepsilon$$

with:

$$\mathbb{E}(\varepsilon) = 0, \ \ Var(\varepsilon) = \sigma^2 \Rightarrow \mathbb{E}(\varepsilon^2) = \sigma^2$$

In regression, we estimate this response variable:

$$\hat{y} = \hat{f}$$

Rewriting the Bias variance decomposition:

$$\Leftrightarrow \mathbb{E}(y - \varepsilon-\hat{f})^2 = Bias(\hat{f})^2 + Var(\hat{f})$$

$$\Leftrightarrow \mathbb{E} [(y-\hat{f})-\varepsilon]^2 = Bias(\hat{f})^2 + Var(\hat{f})$$

$$\Leftrightarrow \mathbb{E} (y-\hat{f})^2 - \mathbb{E}[2\varepsilon(y-\hat{f})] + \mathbb{E}(\varepsilon^2) = Bias(\hat{f})^2 + Var(\hat{f}) $$

$$\Leftrightarrow \mathbb{E} (y-\hat{f})^2 - 2\sigma^2  + \sigma^2 = Bias(\hat{f})^2 + Var(\hat{f}) $$

Which can be written as:

$$\mathbb{E} (y-\hat{f})^2 = Bias(\hat{f})^2 + Var(\hat{f}) + \sigma^2$$

The left hand term is the MSE of the response variable and the predicted value.

# Evaluation Metrics for Regression

## R-Squared

The R Squared value measures how much variability in the dependent variable (Y) can be explained by the model. It can be calculated as:

$$
R^2 = 1 - \frac{\sum \left(y_i - \hat{y}_i \right)^2}{\sum \left(y_i - \bar{y}_i \right)^2}
$$

The R-Squared value lies between 0 to 1 (can technically be smaller than 0), with a value closer to 1 indicating a better fit between prediction and actual value. It is a good measure to determine how well the model fits the dependent variables. However, it does not take into consideration of the overfitting problem. If our regression model has many independent variables (the model is too complicated) it may fit very well to the training data but performs badly for our testing data. That is why Adjusted R Squared is introduced because it will penalize additional independent variables added to the model and adjust the metric to prevent overfitting issues.

## Mean Square Error (MSE)

While the R Squared value is a relative measure of how well the model fits dependent variables, the Mean Square Error (MSE) is an absolute measure of the goodness for the fit.

$$
MSE = \frac{1}{N} \sum_i^N \left(y_i - \hat{y}_i \right)^2
$$

We cannot interpret many insights from one single result but it gives us a real number to compare against other model results and help us select the best regression model.

## Root Mean Square Error (RMSE)

The Root Mean Square Error(RMSE) is the square root of MSE. In some cases, it is used more commonly than MSE for two reasons. Firstly, sometimes the MSE value can be too big to compare easily between models. Secondly, by taking the square root we take the units back to the original units.

## Mean Absolute Error (MAE)

Mean Absolute Error (MAE) is similar to MSE. However, instead of the sum of square of error in MSE, MAE is taking the sum of the absolute value of error:

$$
MAE = \frac{1}{N} \sum_i^N \left| y_i - \hat{y}_i \right|
$$

# Evaluation Metrics for Classification

## Precision

Accuracy of positive predictions. Of all positive predictions (e.g. mails flagged as spam), how many were correct. 

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

TP = Spam emails flagged as spam

FP = Non-spam emails flagged as spam

## Recall (aka. sensitivity, hit rate, TPR)

Ratio of positive instances that are correctly classified by the classifier. Of spam emails, how many are correctly flagged as spam.

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

TP = Spam emails flagged as spam

FN = Spam emails flagged as non-spam

## F1-score

Metric that combines Precision and Recall by taking the harmonic mean (gives more weight to low values than regular mean). 

$$
F_1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}
$$

The classifier will only get a high $F_1$-score if both recall and precision are high.

## Precision/Recall Trade-Off

The $F_1$-score favors classifiers that have similar precision and recall, but this is not always wanted. Two examples:

* If we would train a classifier to flag videos as being safe for kids, we want to reduce the amount of non-safe videos being flagged as safe (FP), so we want a high precision. The trade-off is that we will have a lot of safe videos being falsely rejected (low recall).

* If we would train a classifier to detect thiefs in shops, we want to reduce the amount of thiefs that are flagged as non-thiefs (FN), so we want a high recall. The trade-off is that we will have a lot of non-thiefs being falsely flagged as thiefs (low precision).

Moral of the story: increasing precision reduces recall, and vice versa. If we increase the decision threshold (e.g. higher than 0.5), we are being stricter and we will have more FNs (e.g. flagging safe videos as non-safe), which lowers the recall. However, it lowers the FPs (e.g. flagging non-safe videos as safe), hence a higher precision.

Overall, increasing the threshold will lower the recall (always the case) and increase the precision (sometimes it can go down a bit as well, as it depends on total of predicted TRUE-values).

## PR-curve

The PR (precision-recall) curve plots the precision against recall. It is constructed by calculating the precision and recall for different values of the threshold and is a (most of the time) decreasing function - as precision decreases for decreasing recall.

## Specificity (aka. selectivity, TNR)

Ratio of negative instances that are correctly classified as negative. Of all negative predictions (e.g. mails flagged as non-spam), how many were correct. In other words, what is the percentage of non-spam mails that were flagged as non-spam. 

$$
\text{Specificity} = \frac{TN}{TN+FP}
$$

## False alarm ratio (aka. fall-out, FPR)

Ratio of negative instances that are incorrectly classified as positive. Of all negative predictions (e.g. mails flagged as non-spam), how many were incorrect. In other words, what is the percentage of non-spam mails that were flagged as spam. 

$$
\text{False Alarm Ratio} = \frac{FP}{TN+FP} = 1 - \text{Specificity}
$$

## ROC-curve

The ROC (receiver operating characteristic) curve plots the TPR (Recall) against the FPR (False alarm ratio). This is an increasing function, as the higher the recall (TPR), the more false positives (FPs) the classifier produces. The diagonal of the diagram is the ROC curve of a purely random classifier. A good classifier stays away as far as possible from this line (towards the top left corner).

## AUC

The AUC (area under the curve) is a way of comparing the ROC curve of two classifiers. A perfect classifier will for example have an AUC equal to one. A random classifier will have an AUC equal to 0.5. 

## PR vs. ROC curve

Choose the PR-curve whenever the positive class is rare (e.g. not a lot of spam mails) or when you care more about FPs (e.g. flagging non-spam mails as spam) than FNs (e.g. flagging spam mails as non-spam). The ROC curve might look really good (e.g. high AUC), but this can be due to the fact that there are only a few observations of the positive class. 

## OvR strategy

The OvR (One vs. Rest) or OvA (One vs. All) Strategy can be used when using binary classifiers to perform multiclass classification. For example in MNIST, we might build 10 binary classifiers (0-detector, 1-detector, ...) and use the output with the highest score as the prediction. The 0-detector, for example, is a classifier that outputs whether or not the digit is a zero. 

## OvO strategy

The OvO (One vs. One) Strategy can be used when using binary classifiers to perform multiclass classification. For example in MNIST, we can train a binary classifier for every pair of digits: 0s vs 1s, 0s vs 2s, etc. We run the image through all classifiers and look which class wins the most duels. If there are N classes, we need to train N*(N-1)/2 classifiers. The advantage is that each classifier only needs to be trained on the part of the dataset that contains the two classes it compares. 

This strategy is preferred for algorithms that scale poorly with the size of the training set (e.g. SVMs). For these algorithms OvO is preferred because it is faster to train many classifiers on small training sets than train a few classifiers on large training sets. 










