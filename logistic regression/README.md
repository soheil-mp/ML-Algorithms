# Logistic Regression

<br>

### Description

__Logistic regression__ is an algorithm for binary classification. This algorithm can be divided into 3 parts:
1. Model construction.
2. Cost function.
3. An optimization function (such as Gradient Descent).

<br>

### Model construction

Steps for constructing a logistic regression model:
1. Calculate the output of the following model.
$${z = \hat{w}.\hat{x}+b }$$

2. Apply the sigmoid activation function:
$${ g(z) = \frac{1}{1+e^{-z}} }$$

3. Set a threshold for determining the class number.

<br>

### Cost function

A common cost function for binary classification is (binary) cross-entropy. Its loss function is follows:

$${L(\hat{y}, {y}) = -y.log(\hat{y}) - (1-y).log(1-\hat{y})}$$

For calculating the cost function, take the average of the loss functions over the entire training dataset.

<br>

### Optimization Function

Let's use Gradient Descent as an optimizer.

repeat { 
$${w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\hat{w}, b)}$$
$${b= b - \alpha \frac{\partial}{\partial w_j} J(\hat{w}, b)}$$
$${simultaneously \ update \ w_j \ (for \ j=1, ..., n) \ and \ b}$$
}
