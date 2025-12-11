import numpy as np
import matplotlib.pyplot as plt

def simulate_training_data(num_points=500):
    """
    Simulates fake training data for loss and accuracy.
    
    Parameters:
        num_points (int): Number of points to simulate along the iterations.
        
    Returns:
        iterations (np.array): Simulated iteration numbers.
        loss (np.array): Simulated training loss values.
        accuracy (np.array): Simulated training accuracy values.
    """
    # Simulate iterations from 0 to 80,000 (for example)
    iterations = np.linspace(0, 80000, num_points)
    
    # Simulate a decaying loss curve (exponential decay with a little bit of noise)
    initial_loss = 5.0   # Starting loss value
    decay_constant = 30000.0  # Decay rate parameter; adjust to control the decay speed
    noise_loss = np.random.normal(0, 0.05, num_points)  # small Gaussian noise
    loss = initial_loss * np.exp(-iterations / decay_constant) + noise_loss
    
    # Simulate an increasing accuracy curve reaching very high accuracy near 98%
    # We use a logistic function to simulate the curve
    min_acc = 0.50     # starting accuracy (50%)
    max_acc = 0.98     # final accuracy (98%)
    growth_rate = 0.0002  # growth rate of the logistic function
    midpoint = 40000   # iteration at which accuracy is around halfway between min and max
    
    # Logistic function: accuracy = min + (max - min) / (1 + exp(-growth_rate*(iterations - midpoint)))
    noise_acc = np.random.normal(0, 0.005, num_points)  # very little noise
    accuracy = min_acc + (max_acc - min_acc) / (1 + np.exp(-growth_rate * (iterations - midpoint))) + noise_acc
    accuracy = np.clip(accuracy, 0, 1)  # Ensure accuracy is between 0 and 1
    
    return iterations, loss, accuracy

def plot_metrics(iterations, loss, accuracy):
    """
    Plots training loss and accuracy curves.
    
    Parameters:
        iterations (np.array): Iteration numbers.
        loss (np.array): Loss values.
        accuracy (np.array): Accuracy values.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(iterations, loss, label="Training Loss", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.legend()
    plt.grid(True)
    
    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracy, label="Training Accuracy", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy over Iterations")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    iterations, loss, accuracy = simulate_training_data(num_points=500)
    plot_metrics(iterations, loss, accuracy)

if __name__ == '__main__':
    main()
