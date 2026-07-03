import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set_theme(style="whitegrid")
from tensor import Tensor
from function import ReLU, CrossEntropyWithSoftmax
from optimizer import SGD
from module import Module, Linear

def create_spiral_data(points_per_class, num_classes):
    X = np.zeros((points_per_class * num_classes, 2))
    y = np.zeros(points_per_class * num_classes, dtype='uint8')
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
    return X, y

def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k', alpha=0.8)
    plt.title("Generated Spiral Dataset", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = np.argmax(model.predict(Tensor(grid_points, requires_grad=False)), axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Model Decision Boundary", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    
def animate_decision_boundary(model, X, y, history, filename="decision_boundary_evolution.gif"):
    """
    Creates and saves an animation of the decision boundary evolving over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    epochs = sorted(history.keys())
    
    def update(frame):
        ax.clear()
        epoch = epochs[frame]
        
        model.set_parameters(history[epoch])

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        Z = np.argmax(model.predict(Tensor(grid_points, requires_grad=False)), axis=1)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
        ax.set_title(f"Decision Boundary at Epoch {epoch}", fontsize=16)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        return ax.collections + ax.patches + ax.texts

    ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=100, blit=False)
    
    print(f"\nSaving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=10)
    print("Animation saved.")
    plt.close(fig)

class MLP(Module):
    # TODO
    # Description: Initialize the layers of the Multi-Layer Perceptron.
    # A standard MLP for this task might have:
    # 1. An input linear layer that maps from `num_features` to a hidden dimension (e.g., 100).
    # 2. A non-linear activation function (we'll use ReLU).
    # 3. An output linear layer that maps from the hidden dimension to the `num_classes`.
    #
    # Hint: You should create instances of your `Linear` module here and assign them
    # as attributes of this class (e.g., `self.layer1 = Linear(...)`).
    # The `Module`'s `parameters()` method will automatically find these layers.
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.layer1 = Linear(num_features, 100)
        self.activation = ReLU()
        self.layer2 = Linear(100, num_classes)

    # TODO
    # Description: Define the forward pass of the network.
    # This function specifies how the input data `X` flows through the layers
    # you defined in `__init__`.
    #
    # The sequence of operations should be:
    # 1. Pass the input `X` through your first linear layer.
    # 2. Apply the ReLU activation function to the result of step 1.
    # 3. Pass the result of step 2 through your second (output) linear layer.
    # 4. Return the final output Tensor. This tensor will contain the raw scores (logits)
    #    for each class, which will then be fed into the loss function.
    def forward(self, X: Tensor):
        out = self.layer1(X)
        out = self.activation.apply(out)
        out = self.layer2(out)
        return out


    def predict(self, X):
        return self.forward(X).data
    
if __name__ == "__main__":
    print("Generating dataset...")
    POINTS_PER_CLASS = 100
    NUM_CLASSES = 3
    X, y = create_spiral_data(POINTS_PER_CLASS, NUM_CLASSES)
    print("Data generated.")

    print("Visualizing the raw data...")
    plot_data(X, y)

    model = MLP(num_features=2, num_classes=NUM_CLASSES)

    print("Demonstrating the decision boundary for the UNTRAINED model.")
    plot_decision_boundary(model, X, y)

    print("\nStarting training...")

    X_tensor = Tensor(X)

    num_samples = len(y)
    y_one_hot = np.zeros((num_samples, NUM_CLASSES))
    y_one_hot[np.arange(num_samples), y] = 1
    y_true = Tensor(y_one_hot)

    optimizer = SGD(model.parameters(), lr=0.1)
    
    # with momentum
    #optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Adam 
    #from optimizer import Adam
    #optimizer = Adam(model.parameters(), lr=0.001)


    parameter_history = {}

    EPOCHS = 10000
    # TODO
    # Description: Implement the main training loop.
    # Each iteration of this loop represents one step of model training.
    # The standard procedure for a training step is as follows:
    #
    # 1. Zero Gradients: Before calculating new gradients, you must clear any old ones
    #    from the previous step. Call `optimizer.zero_grad()`.
    #
    # 2. Forward Pass: Get the model's predictions (logits) by passing the input data
    #    `X_tensor` through the model. `y_pred = model.forward(X_tensor)`.
    #
    # 3. Compute Loss: Calculate how "wrong" the model's predictions are by comparing
    #    `y_pred` with the true labels `y_true`. Use your `CrossEntropyWithSoftmax` function.
    #
    # 4. Backward Pass: This is the core of autograd. Call `.backward()` on the loss
    #    tensor. This will automatically compute the gradient of the loss with respect
    #    to every parameter in the model that has `requires_grad=True`.
    #
    # 5. Update Parameters: Tell the optimizer to update the model's parameters using the
    #    gradients computed in the backward pass. Call `optimizer.step()`.
    for epoch in range(EPOCHS+1):
        optimizer.zero_grad()

        y_pred = model.forward(X_tensor)

        loss = CrossEntropyWithSoftmax.apply(y_pred, y_true)

        loss.backward()

        optimizer.step()
        
        if epoch % 250 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS}, Loss: {loss.data.item():.4f}")
            parameter_history[epoch] = [p.data.copy() for p in model.parameters()]

    print("Training complete.")

    animate_decision_boundary(model, X, y, parameter_history)

    print("\nVisualizing the final trained model's decision boundary...")
    plot_decision_boundary(model, X, y)