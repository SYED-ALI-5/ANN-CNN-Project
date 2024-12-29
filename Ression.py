import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    # Step 1: Load California Housing Dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)  # Make y a column vector

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class CustomRegressionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRegressionModel, self).__init__()
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(output_size, requires_grad=True)
       

    def forward(self, x):
        hidden = torch.relu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output

def train_model(model, X_train, y_train, batch_size, epochs, loss_fn, optimizer):
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            model.train()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return epoch_losses

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mae = torch.mean(torch.abs(y_test - y_pred))
        mse = torch.mean((y_test - y_pred) ** 2)
        ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
        ss_residual = torch.sum((y_test - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
    return mae.item(), mse.item(), r2.item()

# Main script
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Model Parameters
input_size = X_train.shape[1]
hidden_size = 16
output_size = 1
learning_rate = 0.01
batch_size = 64
epochs = 100

# Model, Loss Function, Optimizer
model = CustomRegressionModel(input_size, hidden_size, output_size)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=learning_rate)

# Train Model
losses = train_model(model, X_train, y_train, batch_size, epochs, loss_fn, optimizer)

# Plot Loss Curve
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Evaluate Model
mae, mse, r2 = evaluate_model(model, X_test, y_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
