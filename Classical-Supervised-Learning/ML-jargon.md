# Bias-Variance

## Bias

$$\text{Bias}(\hat{f}) = \mathbb{E}(\hat{f}) - f$$

## Variance

$$\text{Var}(\hat{f}) = \mathbb{E}\left[\left(\hat{f} - \mathbb{E}(\hat{f})\right)^2\right]$$

## Bias-Variance Decomposition

The response variable `y` can be defined as a true function `f` plus noise `\epsilon`. This can be written as follows:

$$y = f + \varepsilon$$

with:

$$E(\varepsilon) = 0, \: Var(\varepsilon) = \sigma^2 \Rightarrow E(\varepsilon^2) = \sigma^2$$

In regression, we estimate this response variable:

$$\hat{y} = \hat{f}$$

The Bias variance decomposition can be written as:

$$E(f-\hat{f})^2 = Bias(\hat{f})^2 + Var(f)$$

$$\Leftrightarrow E(y - \varepsilon-\hat{f})^2 = Bias(\hat{f})^2 + Var(f)$$

$$\Leftrightarrow E [(y-\hat{f})-\varepsilon]^2 = Bias(\hat{f})^2 + Var(f)$$

$$\Leftrightarrow E (y-\hat{f})^2 - E[2\varepsilon(y-\hat{f})] + E(\varepsilon^2) = Bias(\hat{f})^2 + Var(f) $$

$$\Leftrightarrow E (y-\hat{f})^2 - 2\sigma^2  + \sigma^2 = Bias(\hat{f})^2 + Var(f) $$

Which can be written as:

$$E (y-\hat{f})^2 = Bias(\hat{f})^2 + Var(f) + \sigma^2$$

The left hand term is the MSE as defined in subsection 1.1 (with the y-values). The above formula proves that if one substracts the squared bias and variance for this data set from the above defined MSE value, this results in $\sigma^2$. The latter being equal to 1 in this case. To show this, one could also change the sigma value in the definition of the noise. If one would for example use a $\sigma = 2$, the difference would be 4.

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










