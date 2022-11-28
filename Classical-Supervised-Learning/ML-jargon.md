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

FP = Spam emails flagged as non-spam

## F1-score

Metric that combines Precision and Recall by taking the harmonic mean (gives more weight to low values than regular mean). 

$$
F_1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}
$$
