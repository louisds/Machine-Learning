# Support Vector Machine (SVM)

Support Vector Machine (SVM) is a simple ML algorithm that can be used for text classification tasks. In short: The objective of the SVM algorithm is to find a hyperplane (decision boundary) in an N-dimensional space (N being the number of features, e.g. word occurences) that distinctly classifies the data points. The optimal hyperplane is the plane that has the maximum margin, i.e the maximum distance between data points of both classes, so that future data points can be classified with more confidence. 

## What's in a name?

So where does the name SVM comes from? Support vectors are the data points that are closer to our hyperplane and influence the position and orientation of it. Using these support vectors, we construct our SVM, as they maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. 

## Difference with Logistic Regression? 

In what way is this different than Logistic Regression? In binary logistic regression, we feed the output of the linear function (the log-odds) through a sigmoid function, leading to values within the range of [0,1]. In the binary case, we assign the label 1 if the output is greater than a threshold value (0.5), else we assign it a label 0. In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values ([-1,1]) which acts as the margin.

## Cost Function and Gradient Updates

As mentioned above, we are looking to maximize the margin between the data points and the hyperplane. The loss function that performs this optimization is called the "hinge loss":

$$ 
c(x, y, w) = \begin{cases} 
            0, \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \text{if} \ y x^T w \geq 1 \\
             \\
            (1 - y x^T w) \  \ \ \text{else} \\
       \end{cases} 
$$

which can also be written as:

$$ 
c(x, y, w) = (1 - y x^T w)_+
$$

So the loss is 0 if the predicted value and the actual value are of the same sign. If they are not, we then calculate the loss value. To balance the margin maximization and loss, we also add a regularization parameter $\lambda$ to the cost function. This gives:

$$
\mathcal{L} = \sum_{i=1}^{n} (1 - y_i x_i^T w)_{+} \ + \lambda \lVert w \rVert
$$

which has to be minimized with respect to $w$. Taking the derivative, the parameter update can be split into two situations. When there is no misclassification, i.e our model correctly predicts the class of our data point, we only have to update the gradient from the regularization parameter:

$$
w \leftarrow w - \eta (2 \lambda w)
$$

When there is a misclassification, i.e our model make a mistake on the prediction of the class of our data point, we include the loss along with the regularization parameter to perform gradient update:

$$
w \leftarrow w + \eta (y_i x_i - 2 \lambda w)
$$

