# Evaluation Metrics

## Precision

Accuracy of positive predictions. Of all positive predictions (e.g. mails flagged as spam), how many were correct. 

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

TP = Spam emails flagged as spam

FP = Non-spam emails flagged as spam

## Recall (aka. sensitivity, TPR)

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

## ROC-curve

The ROC (receiver operating characteristic) curve










