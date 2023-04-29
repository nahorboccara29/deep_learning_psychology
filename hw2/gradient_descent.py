import numpy as np
import matplotlib.pyplot as plt

# Question 1, 2
'''
The main difference between linear regression and logistic regression lies in the type of problem they are designed to solve and the way they model the
relationship between the independent (input) and dependent (output) variables. Linear regression is used for continuous output variables, whereas 
logistic regression is used for binary or categorical output variables.

Linear regression models the relationship between the independent variables and the dependent variable as a linear function. It aims to minimize the sum
of squared errors between the predicted and actual values of the dependent variable. Linear regression can predict continuous values, but it is not
suitable for predicting probabilities, as its output can range from negative infinity to positive infinity, and probabilities must be constrained between
0 and 1.

Logistic regression, on the other hand, is designed to model the probability of an event occurring given a set of input variables. It uses the logistic
function (also called the sigmoid function) to transform the linear combination of input variables into a probability value between 0 and 1. The logistic
function maps any real-valued input to a value between 0 and 1, which makes it ideal for modeling probabilities.

Using linear regression to solve a probabilistic question can lead to several issues:

1. Predicted values outside the range of 0 and 1: Since linear regression does not constrain its output between 0 and 1, the predicted probabilities may
not be valid (i.e., less than 0 or greater than 1).

2. Inappropriate loss function: Linear regression uses the mean squared error (MSE) as its loss function. However, for probabilistic questions, the more
appropriate loss function is cross-entropy loss, which measures the divergence between predicted probabilities and true probabilities.

3. Non-linearity of probabilities: Probabilities exhibit non-linear behavior, especially near the boundaries of 0 and 1. Linear regression, being a
linear model, cannot capture this non-linearity, which may result in poor performance for probabilistic questions.

In summary, logistic regression is more appropriate for probabilistic questions because it models the relationship between input variables and output
probabilities using a logistic function, which provides predictions between 0 and 1, better handles the non-linearity of probabilities, and uses a
more appropriate loss function for such problems.
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x

x = np.linspace(-10, 10, 1000)
y_sigmoid = sigmoid(x)
y_linear = linear(x)

plt.plot(x, y_sigmoid, label = 'Sigmoid')
plt.plot(x, y_linear, label = 'Linear')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sigmoid vs Linear Function')
plt.grid()
plt.show()

# Question 3
def predict_logistic_regression(w, b, x):
    z = np.dot(x, w) + b
    y_pred = sigmoid(z)
    return y_pred

# Question 4
def gradient_w(x, y, y_hat):
    N, _ = x.shape
    y = y[:, np.newaxis]
    y_hat = y_hat[:, np.newaxis]
    error = y_hat - y
    gradient = np.dot(x.T, error) / N
    return gradient

# Question 5
'''
In words, the change in 'w' during gradient descent is an adjustment of the weights to minimize the discrepancy between the predicted probabilities and the
true labels. This change helps the model learn the underlying relationship between the input features and the output labels. The gradient represents the
direction in which the weights should be updated to reduce the loss most effectively.

After an infinite number of iterations, the composition of w would converge to an optimal set of weights that minimize the loss function. However, in practice,
we usually stop the training process after a certain number of iterations or when the improvement in the loss function becomes negligible. It is important to
note that logistic regression can find a global minimum for the loss function, given that the cross-entropy loss is convex.

The magnitude of the change in w during gradient descent depends on both the input features x and the discrepancy between the predicted probabilities and the
true labels (i.e., (y - y_hat)).

For input features x: Larger input values will result in larger changes in w, as the gradient calculation involves multiplying the input features by the error.
In other words, if an input feature has a more significant effect on the output, the model will update the corresponding weight with a larger step during
gradient descent.

For discrepancy (y - y_hat): If the predicted probability is far from the true label, the gradient will be larger, resulting in a larger update to the weights.
This means that samples that the model struggles to predict correctly will have a more significant impact on the weight updates during gradient descent.

Regarding the generalization to training neural networks and the concept of perceptual narrowing, the composition of trained weights can be related to the
"other-race-effect." Perceptual narrowing is the process by which an individual's ability to discriminate between different stimuli, such as faces or speech
sounds, becomes more refined and specialized over time. For example, the "other-race-effect" refers to the phenomenon where individuals are better at
recognizing faces from their own racial or ethnic group compared to faces from other racial or ethnic groups.

As a neural network is trained, the weights are updated to capture the relevant features that help discriminate between different classes. This process can be
seen as a form of perceptual narrowing, where the network becomes specialized in recognizing certain patterns or features in the data. In the context of the
"other-race-effect," if a neural network is trained predominantly on faces from one racial or ethnic group, the weights will be optimized to recognize and
discriminate between faces within that group. This can lead to reduced performance when the network is presented with faces from other racial or ethnic groups,
as the network's weights have been fine-tuned to recognize features specific to the training group.
'''

# Question 6
def gradient_b(y, y_hat):
    N = len(y)
    error = y_hat - y
    gradient = np.sum(error) / N
    return gradient

# Question 7
'''
In logistic regression, we have a weight vector 'w' and a bias term 'b'. The weight vector w represents how important each input feature is in determining the
output, while the bias term 'b' helps in adjusting the overall output. When we update the bias term 'b', we are essentially shifting the output of the model to
better match the true labels of the samples.

In logistic regression, we are trying to classify data points into one of two classes (e.g., positive or negative). The decision boundary is a line (or a plane,
in higher-dimensional spaces) that separates the two classes. Data points on one side of the decision boundary are predicted to belong to one class, while data
points on the other side are predicted to belong to the other class.

In simpler terms, updating the bias 'b' helps to shift the decision boundary to a position where it can better separate the two classes in the data. The change in
'b' depends on the difference between the predicted probabilities and the true labels (i.e., (y - y_hat)). When this difference is larger, it means that the model
is not predicting the samples correctly, and the bias term will be updated more to improve the predictions.

So, in summary:

The change in 'b' helps shift the decision boundary, which is the line that separates the two classes in the data.
The decision boundary is adjusted to improve the model's predictions.
The magnitude of the change in b depends on the difference between the predicted probabilities and the true labels. Larger differences will cause larger updates
to the bias term.
'''

# Question 8
def generate_data(N):
    # Parameters for the two normal distributions
    mu_1 = np.array([1, 1])
    mu_2 = np.array([-1, -1])
    sigma = 1

    # Generate N samples from each distribution
    samples_1 = np.random.normal(mu_1, sigma, size = (N, 2))
    samples_2 = np.random.normal(mu_2, sigma, size = (N, 2))

    # Combine the samples from both distributions
    samples = np.vstack((samples_1, samples_2))

    # Create corresponding labels
    labels_1 = np.ones(N)
    labels_2 = np.zeros(N)
    labels = np.hstack((labels_1, labels_2))

    return samples, labels

# Question 9
def logistic_regression_step(X, y, w, b, learning_rate):
    # Calculate the predicted probabilities using the logistic function
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z))

    # Calculate the mean loss (cross-entropy loss)
    mean_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    # Calculate the gradients for w and b
    error = y_hat - y
    gradient_w = np.dot(X.T, error) / len(X)
    gradient_b = np.mean(error)

    # Update the weights and bias using the gradients
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    return w, b, mean_loss

# Question 10
def logistic_regression_train(X, y, w, b, learning_rate, num_iterations):
    loss_history = []

    for _ in range(num_iterations):
        w, b, mean_loss = logistic_regression_step(X, y, w, b, learning_rate)
        loss_history.append(mean_loss)

    return w, b, loss_history

# Question 11
num_iterations = 1000
w = np.random.randn(2)  # Initialize random weights
b = np.random.randn()   # Initialize random bias
samples, labels = generate_data(num_iterations)
learning_rate = 0.01

w, b, loss_history = logistic_regression_train(samples, labels, w, b, learning_rate, num_iterations)
plt.plot(loss_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Question 12
iteration_list = [1, 10, 100, 1000]
N = 100
w = np.random.randn(2)  # Initialize random weights
b = np.random.randn()   # Initialize random bias
samples, labels = generate_data(N)
learning_rate = 0.01

def plot_decision_boundary(X, y, w, b, title):
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'viridis')
    x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_values = -(w[0] * x_values + b) / w[1]
    plt.plot(x_values, y_values, 'r--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.show()

# Train for different numbers of iterations and plot the decision boundary

for iterations in iteration_list:
    w, b, _ = logistic_regression_train(samples, labels, w, b, learning_rate, iterations)
    plot_decision_boundary(samples, labels, w, b, f'Decision Boundary after {iterations} iterations')

# Question 13
'''
Yes, there is a correlation between the size of the change in the classifier and the size of the loss.

When the loss is high, it indicates that the current classifier is not performing well on the training data. The gradient of the loss function with respect to the
model parameters (weights and bias) will be large, which leads to a significant update in the parameters during gradient descent. As a result, the classifier
experiences a larger change.

On the other hand, when the loss is low, it means the classifier is performing well on the training data. In this case, the gradient of the loss function with
respect to the model parameters will be small. Consequently, the updates to the parameters during gradient descent will be smaller, causing a smaller change in the
classifier.

As the training progresses and the classifier gets better at minimizing the loss, the size of the updates to the model parameters decreases. This is because the
gradient of the loss function becomes smaller as the classifier approaches the optimal solution. This correlation between the size of the change in the classifier
and the size of the loss helps ensure that the classifier converges to a solution that minimizes the loss function.
'''
