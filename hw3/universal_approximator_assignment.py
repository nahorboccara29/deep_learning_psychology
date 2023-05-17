import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Question 1
class MultiLayerNN(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim):
        super(MultiLayerNN, self).__init__()
        
        # Check if number of hidden layers is 0 (linear regression)
        if num_hidden_layers == 0:
            self.model = nn.Sequential(
                nn.Linear(1, 1)
            )
        else:
            # Initialize the list of layers
            layers = []
            
            # Input layer
            layers.append(nn.Linear(1, hidden_dim))
            layers.append(nn.ReLU())

            # Add hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))

            # Create the model using the layers list
            self.model = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.model(x)

# Question 2
mlp = MultiLayerNN(2, 32)

# Create an MSELoss object
criterion = torch.nn.MSELoss()

# Create an SGD optimizer with a learning rate of 0.01
optimizer = optim.SGD(mlp.parameters(), lr=0.01)

# Question 3
def sample_data(batch_size, noise_std_dev, target_function):
    # Sample random x values from the range [-π, π]
    x = np.random.uniform(-np.pi, np.pi, size=(batch_size, 1))

    # Calculate the corresponding y values using the target function
    y = target_function(x)

    # Sample random noise from the normal distribution with mean 0 and standard deviation σ
    noise = np.random.normal(0, noise_std_dev, size=(batch_size, 1))

    # Add the noise to the y values
    y_noisy = y + noise

    # Convert x and y_noisy to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_noisy_tensor = torch.tensor(y_noisy, dtype=torch.float32)

    return x_tensor, y_noisy_tensor

# Question 4
def train_model(model, loss_function, optimizer, batch_size, sigma, target_function, epochs):
    for epoch in range(epochs):
        # Sample a training batch
        x_batch, y_noisy_batch = sample_data(batch_size, sigma, target_function)

        # Forward pass
        y_pred = model(x_batch)

        # Compute the loss
        loss = loss_function(y_pred, y_noisy_batch)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Print the loss for this epoch
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model

# Example usage:
my_target_function = lambda x: x**2  # Define the target function
trained_model = train_model(mlp, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100)

# Question 5
def training_loop(model, loss_function, optimizer, batch_size, sigma, target_function, epochs, iterations):
    for iteration in range(iterations):
        print(f'Iteration {iteration + 1}')
        model = train_model(model, loss_function, optimizer, batch_size, sigma, target_function, epochs)
    return model

# Example usage:
trained_model = training_loop(mlp, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=5)

# Question 6
def target_function(x):
    return x**2

# Generate a range of x values for plotting
x_range = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)

# Calculate the true y values based on the target function
y_true = target_function(x_range)

# Convert x_range to a PyTorch tensor for model evaluation
x_range_tensor = torch.tensor(x_range, dtype=torch.float32)

# Evaluate the trained model on the x_range_tensor
y_pred = trained_model(x_range_tensor).detach().numpy()

# Plot the target function and the learned function
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
plt.plot(x_range, y_pred, label='Learned Function', linestyle='-', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Target Function and Learned Function')
plt.legend()
plt.show()

# Question 7

# a
batch_sizes = [1, 5, 10]

for batch_size in batch_sizes:
    print(f'Training with batch size: {batch_size}')
    
    # Initialize the model, loss function, and optimizer
    mlp = MultiLayerNN(2, 32)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.01)

    # Train the model using the specified batch size
    trained_model = training_loop(mlp, criterion, optimizer, batch_size=batch_size, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

    # Compare the target function and the learned function
    y_pred = trained_model(x_range_tensor).detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
    plt.plot(x_range, y_pred, label='Learned Function', linestyle='-', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of Target Function and Learned Function (Batch Size: {batch_size})')
    plt.legend()
    plt.show()

'''
Impact of batch size:

1. Training speed: As the batch size increases, each epoch takes longer to complete because the model processes more data points at once. However, the overall training
 time might decrease, as larger batch sizes generally require fewer epochs to converge.
2. Memory usage: Larger batch sizes require more memory to store intermediate values during training. This can be a limiting factor on hardware with limited memory, such
 as GPUs.
3. Convergence: Smaller batch sizes often lead to noisier updates in the optimization process, which can help the model escape local minima and explore the loss landscape
 more thoroughly. However, this can also lead to less stable convergence. Larger batch sizes result in more stable updates but may get stuck in local minima more
 easily.

Catastrophic forgetting in neural networks refers to the phenomenon where a model quickly loses its ability to perform a previously learned task when learning a new
task. This is particularly relevant in the context of online learning, where the model is trained sequentially on different tasks. In our case, a smaller batch size
(e.g., 1) is more likely to cause catastrophic forgetting, as the model updates its parameters based on individual data points, which may result in overwriting
previously learned information. On the other hand, larger batch sizes (e.g., 10) provide a more stable update, which can help mitigate catastrophic forgetting.

In summary, the choice of batch size has a significant impact on the training process, memory usage, and convergence behavior. It's essential to strike a balance
between these factors when selecting a batch size for a given problem.
'''

# b
learning_rates = [1e-5, 1e-3, 0.1, 1]

for lr in learning_rates:
    print(f'Training with learning rate: {lr}')

    # Initialize the model, loss function, and optimizer
    mlp = MultiLayerNN(2, 32)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(mlp.parameters(), lr=lr)

    # Train the model using the specified learning rate
    trained_model = training_loop(mlp, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

    # Compare the target function and the learned function
    y_pred = trained_model(x_range_tensor).detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
    plt.plot(x_range, y_pred, label='Learned Function', linestyle='-', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of Target Function and Learned Function (Learning Rate: {lr})')
    plt.legend()
    plt.show()

'''
Impact of learning rate:

1. Convergence speed: A higher learning rate can lead to faster convergence, as the model's parameters are updated more aggressively. However, too high a learning rate
may result in overshooting the optimal solution and cause oscillations in the loss landscape.
2. Stability: A lower learning rate results in more stable and precise updates but may take much longer to converge. In some cases, a learning rate that is too low
might also cause the model to get stuck in local minima or plateaus in the loss landscape.
3. Optimal solution: The choice of learning rate can significantly impact the quality of the final solution. A learning rate that is too high might cause the model to
miss the optimal solution, while a learning rate that is too low might cause the model to settle for suboptimal solutions.

In the given example, a learning rate of 1e-5 may be too small, resulting in slow convergence, while a learning rate of 1 may be too large, causing the model to
overshoot the optimal solution. The learning rates 1e-3 and 0.1 may be more appropriate for this problem, but the optimal learning rate will depend on the specific
problem and model architecture.

In summary, the learning rate is a crucial hyperparameter that affects the speed, stability, and quality of the model's convergence. Choosing an appropriate learning
rate is essential for training a neural network effectively.
'''

# c
hidden_dims = [1, 5, 10, 20]

for hidden_dim in hidden_dims:
    print(f'Training with {hidden_dim} hidden dimensions')

    # Initialize the model, loss function, and optimizer
    mlp = MultiLayerNN(2, hidden_dim)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.01)

    # Train the model using the specified number of hidden dimensions
    trained_model = training_loop(mlp, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

    # Compare the target function and the learned function
    y_pred = trained_model(x_range_tensor).detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
    plt.plot(x_range, y_pred, label='Learned Function', linestyle='-', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of Target Function and Learned Function (Hidden Dimensions: {hidden_dim})')
    plt.legend()
    plt.show()

'''
Impact of hidden dimensions:

1. Model complexity: A higher number of hidden dimensions increases the model's complexity and capacity to learn more intricate relationships in the data. This can
lead to better performance, especially when the target function is complex.
2. Risk of overfitting: Increasing the number of hidden dimensions can increase the risk of overfitting, as the model may fit noise in the training data instead of
learning the underlying pattern. This is especially true when the number of training data points is limited.
3. Training speed: A higher number of hidden dimensions can increase the training time, as more parameters need to be updated during each optimization step.

In the given example, using just 1 hidden dimension might not be enough for the model to learn the target function effectively. As the number of hidden dimensions
increases, the model becomes more capable of learning the underlying pattern. However, it is essential to strike a balance between model complexity and the risk of
overfitting, especially when working with limited training data.

In summary, the number of hidden dimensions is a crucial hyperparameter that affects the model's complexity, performance, and training speed. Choosing an appropriate
number of hidden dimensions is essential for training a neural network effectively.
'''

# d
class MultiLayerNN_Sigmoid(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim):
        super(MultiLayerNN_Sigmoid, self).__init__()
        layers = []
        input_dim = 1
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Sigmoid())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Train the model using the Sigmoid activation function
mlp_sigmoid = MultiLayerNN_Sigmoid(2, 10)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(mlp_sigmoid.parameters(), lr=0.01)

trained_model_sigmoid = training_loop(mlp_sigmoid, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

# Compare the target function and the learned function with the Sigmoid activation function
y_pred_sigmoid = trained_model_sigmoid(x_range_tensor).detach().numpy()
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
plt.plot(x_range, y_pred_sigmoid, label='Learned Function (Sigmoid)', linestyle='-', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Target Function and Learned Function (Sigmoid Activation)')
plt.legend()
plt.show()

'''
i. Changes in smoothness of the learned function: The Sigmoid activation function results in a smoother learned function compared to the ReLU activation function. This 
s because the Sigmoid function is differentiable everywhere and has a continuous, smooth transition between output values.

ii. Comparison between ReLU(x) and Sigmoid(x):

ReLU(x) = max(0, x): ReLU is a piecewise linear function, with a sharp transition at x=0. It is non-linear, but the gradient is either 0 or 1. This makes it
computationally efficient and helps alleviate the vanishing gradient problem.
Sigmoid(x) = 1 / (1 + exp(-x)): The Sigmoid function is a smooth, differentiable function that maps input values to the range (0, 1). It has a smooth transition
between output values, but it can suffer from the vanishing gradient problem, as the gradient becomes very small for large positive or negative input values.
The smoother learned function when using the Sigmoid activation function is a result of the smooth, continuous nature of the Sigmoid function, which allows the model
to learn a smoother approximation of the target function.

iii. For a very large X value (X > 3), the gradient of the Sigmoid function becomes very small (approaching 0), while the gradient of the ReLU function remains 1. This
difference affects the step size for each optimization step, as the gradients are used to update the model's parameters. With a small gradient, the Sigmoid activation
function will result in smaller updates during optimization, which may slow down the training process. On the other hand, the ReLU activation function maintains a
constant gradient of 1 for positive input values, which can help alleviate the vanishing gradient problem and allow for faster training.
'''

# e
num_hidden_layers_list = [0, 1, 2, 3, 4, 8]

for num_hidden_layers in num_hidden_layers_list:
    print(f'Training with {num_hidden_layers} hidden layers')

    # Initialize the model, loss function, and optimizer
    mlp = MultiLayerNN(num_hidden_layers, 10)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(mlp.parameters(), lr=0.01)

    # Train the model using the specified number of hidden layers
    trained_model = training_loop(mlp, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

    # Compare the target function and the learned function
    y_pred = trained_model(x_range_tensor).detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
    plt.plot(x_range, y_pred, label='Learned Function', linestyle='-', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of Target Function and Learned Function (Hidden Layers: {num_hidden_layers})')
    plt.legend()
    plt.show()

'''
i. As you increase the number of hidden layers, the model becomes more expressive and capable of learning complex patterns in the data. However, increasing the number
of hidden layers also increases the risk of overfitting, especially when the training data is limited. Additionally, as the number of hidden layers increases, the
model becomes deeper, which can make training more difficult due to issues like the vanishing or exploding gradient problem.

ii. When using 8 hidden layers, the model is much deeper and has a higher capacity to learn intricate relationships in the data. However, this can also increase the
risk of overfitting and make the training process more challenging. In some cases, using a deep model like this might require more sophisticated optimization
techniques or regularization methods to ensure effective learning.

Regarding the gradient size when composing a large number of functions, if the derivative of each function is in the range (-1, 1) and slightly smaller than 1, the
gradient size will generally decrease as more functions are composed. This is due to the chain rule:

(f_n(f_(n-1)(…(x))))' = f_n'(…)⋅f_(n-1)'(…)⋅…

When you multiply a series of numbers that are slightly smaller than 1, the product will become smaller as you include more factors. This is the vanishing gradient
problem, which can make training deep neural networks more challenging. When gradients become very small, updates to the model's parameters during optimization will
also be very small, which can lead to slow convergence or getting stuck in suboptimal solutions.
'''

# Question 8
class ResidualLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualLayer, self).__init__()
        self.lin_1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.lin_1(x)
        out = self.relu(out)
        out = self.lin_2(out)
        out += identity
        return out

# Question 9
class ResNet(nn.Module):
    def __init__(self, num_residual_layers, hidden_dim):
        super(ResNet, self).__init__()
        
        # First linear layer to set up the dimensions
        self.input_layer = nn.Linear(1, hidden_dim)
        
        # Residual layers
        residual_layers = [ResidualLayer(hidden_dim) for _ in range(num_residual_layers)]
        self.residual_layers = nn.Sequential(*residual_layers)
        
        # Last linear layer to project back to 1-dimensional output
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x

# Question 10
# Train the ResNet model
resnet = ResNet(3, 10)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01)

trained_resnet = training_loop(resnet, criterion, optimizer, batch_size=10, sigma=0.1, target_function=my_target_function, epochs=100, iterations=1)

# Compare the target function and the learned function
y_pred_resnet = trained_resnet(x_range_tensor).detach().numpy()
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_true, label='Target Function', linestyle='--', color='blue')
plt.plot(x_range, y_pred_resnet, label='Learned Function (ResNet)', linestyle='-', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Target Function and Learned Function (ResNet with 3 Residual Layers)')
plt.legend()
plt.show()

'''
a. In total, there are 7 linear layers in the model: 1 input layer, 1 output layer, and 3 residual layers, each containing 2 linear layers.

b. The function seems to converge, although it may not be perfect. Convergence can be affected by various factors such as learning rate, batch size, and number of
epochs.

c. Comparing the performance of the ResNet with an MLP having the same number of linear layers (7) might reveal that the ResNet performs better, especially when
dealing with deeper architectures. The residual connections in ResNet help to mitigate the vanishing gradient problem, which can make training deep networks more
stable and efficient.

d. The residual trick likely affected the magnitude of the gradient positively. By adding the input directly to the output in each residual layer, gradients can more
easily backpropagate through the network. This helps to alleviate the vanishing gradient problem, allowing the model to learn more effectively in deeper architectures.
The residual connections provide a direct path for gradients to flow, which can improve the training process and enable the model to learn more complex functions.
'''

# Question 11
'''
a. In a neural network, information from different processing levels is combined to form a single representation. Each layer of the network applies a transformation to
the input data, allowing the model to learn and represent more complex features as it progresses through the layers. In the case of residual networks, the residual
connections help to preserve information from earlier layers by adding it directly to the output of a later layer.

b. The depth of a residual layer can be defined as the number of transformations applied to the input within that layer, which is typically 2 for a basic residual
layer (i.e., the two linear layers). This is considering the fact that the input is transformed twice before being combined with the identity mapping (the input
itself) to produce the output.

c. The depth of a linear layer within a residual layer can be seen as 1. Each linear layer represents a single transformation applied to the input, and contributes to
the overall depth of the residual layer.

d. When comparing to behavior or cognition, the information in a deep layer can be characterized as highly processed. As the input data progresses through the layers,
it is transformed and combined with information from other layers, allowing the network to learn and represent more complex and abstract features. In the context of
neuroscience, this can be seen as analogous to the hierarchical processing of information in the brain, where higher-level cognitive processes build upon the output of
lower-level sensory and perceptual processes.
'''