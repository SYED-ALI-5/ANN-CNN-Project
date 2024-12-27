# SP23-BAI-050
# SYED AHMAD ALI

# PyTorch ANN's CLASSIFICATION
# DataSet: MNIST (Modified National Institute of Standards and Technology)

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as raw_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset ,DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def CompilingData():
    # Loading the dataset
    dataset = raw_dataset.MNIST(root="./data", train=True, download=True)

    # Extract images (X) and labels (y) from the dataset
    X = dataset.data
    y = dataset.targets

    """
    print("X Shape:", X.shape)
    print("X:", X)
    """

    # Converting them to NumPy arrays
    X = X.numpy()
    y = y.numpy()

    # Normalizing the data
    X = X / 255.0
    X = (X - 0.5) / 0.5

    # Flatten the images (convert 28x28 images into 784-dimensional vectors)
    X = X.reshape(X.shape[0], -1)

    """
    print("X_Flaten Shape:", X.shape)
    print("X_Flaten:", X)
    """

    # Converting to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    """
    print("X_tensor Shape: ", X.shape)
    print("y_tensor Shape: ", y.shape)
    """
    return X, y

def train_model(model, data, batch_size, n_epoch, loss_fn, optimizer):
    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    epoch_losses = []  # To store average loss for each epoch
    batch_losses = []  # To store losses for each batch in the epoch
    
    for epoch in range(n_epoch):
        for x_batch, y_batch in train_loader:
            model.train()
            yhat = model(x_batch)
            loss = loss_fn(yhat, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())  # Store batch loss
        
        # Compute average loss for the epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)  # Store epoch-level loss
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    return epoch_losses

def evaluate_with_metrics(model, data, batch_size, loss_fn):
    test_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in test_loader:
            yhat = model(x_batch)
            loss = loss_fn(yhat, y_batch)
            total_loss += loss.item()
            
            # Calculate predictions
            _, predictions = torch.max(yhat, 1)
            all_predictions.extend(predictions.cpu().numpy())  # Store predictions
            all_labels.extend(y_batch.cpu().numpy())  # Store true labels
    
    # Calculate accuracy
    accuracy = (sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)) * 100
    avg_loss = total_loss / len(test_loader)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report for precision, recall, and F1-score
    report = classification_report(all_labels, all_predictions, digits=4)
    
    return accuracy, avg_loss, cm, report

class CustomClassificationModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        super(CustomClassificationModel, self).__init__()
        self.W1 = torch.randn(input_size, hidden_size1, requires_grad=True)
        self.b1 = torch.randn(hidden_size1, requires_grad=True)
        self.W2 = torch.randn(hidden_size1, hidden_size2, requires_grad=True)
        self.b2 = torch.randn(hidden_size2, requires_grad=True)
        self.W3 = torch.randn(hidden_size2, hidden_size3, requires_grad=True)
        self.b3 = torch.randn(hidden_size3, requires_grad=True)
        self.W4 = torch.randn(hidden_size3, hidden_size4, requires_grad=True)
        self.b4 = torch.randn(hidden_size4, requires_grad=True)
        self.W5 = torch.randn(hidden_size4, hidden_size5, requires_grad=True)
        self.b5 = torch.randn(hidden_size5, requires_grad=True)
        self.W6 = torch.randn(hidden_size5, output_size, requires_grad=True)
        self.b6 = torch.randn(output_size, requires_grad=True)
        
        torch.nn.init.xavier_normal_(self.W1)
        torch.nn.init.xavier_normal_(self.W2)
        torch.nn.init.xavier_normal_(self.W3)
        torch.nn.init.xavier_normal_(self.W4)
        torch.nn.init.xavier_normal_(self.W5)
        torch.nn.init.xavier_normal_(self.W6)
    
    def forward(self, input_data):
        hidden1 = torch.relu((input_data @ self.W1 + self.b1))
        hidden2 = torch.relu((hidden1 @ self.W2 + self.b2))
        hidden3 = torch.relu((hidden2 @ self.W3 + self.b3))
        hidden4 = torch.relu((hidden3 @ self.W4 + self.b4))
        hidden5 = torch.relu((hidden4 @ self.W5 + self.b5))
        output = hidden5 @ self.W6 + self.b6

        return output

X, y = CompilingData()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
print("X_train Shape: ", X_train.shape)
print("y_train Shape: ", y_train.shape)
print("X_test Shape: ", X_test.shape)
print("y_test Shape: ", y_test.shape)
"""

# Creating Tensor dataset
input = 784
hidden_size1 = 200
hidden_size2 = 170
hidden_size3 = 140
hidden_size4 = 90
hidden_size5 = 50
output_size = 10
learning_rate = 0.01
batch_size = 50
epochs = 20

model = CustomClassificationModel(input_size=input, hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, hidden_size4=hidden_size4, hidden_size5=hidden_size5, output_size=output_size)
train_dataset = TensorDataset(X_train, y_train)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2, model.W3, model.b3, model.W4, model.b4, model.W5, model.b5, model.W6, model.b6],lr=learning_rate)
epoch_loses = train_model(model=model, data=train_dataset, batch_size=batch_size, n_epoch=epochs, loss_fn=loss_fn, optimizer=optimizer)

plt.plot(range(epochs), epoch_loses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Testing the Model
test_dataset = TensorDataset(X_test, y_test)
accuracy, avg_loss, cm, report = evaluate_with_metrics(model=model, data=test_dataset, batch_size=batch_size, loss_fn=loss_fn)

# Displaying results
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {avg_loss:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Ploting confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(report)

# Displaying some images and their labels
"""
images_list, labels_list = [], []

# Loop through batches until we gather 144 samples
for batch_images, batch_labels in train_loader:
    images_list.append(batch_images)
    labels_list.append(batch_labels)
    if len(images_list) * batch_size >= 144:  # Stop once we have enough samples
        break

# Combine collected batches into a single tensor
images = torch.cat(images_list, dim=0)[:144]
labels = torch.cat(labels_list, dim=0)[:144]

# Plot the first 144 images
fig, axes = plt.subplots(12, 12, figsize=(24, 24))  # Create a 12x12 grid of subplots
for i in range(144):  # Loop through the first 144 images
    row, col = divmod(i, 12)  # Calculate row and column indices for the grid
    axes[row, col].imshow(images[i].numpy().squeeze(), cmap='gray')  # Plot the image
    axes[row, col].set_title(f"Label: {labels[i].item()}", fontsize=8)  # Set the label
    axes[row, col].axis('off')  # Turn off axes for better clarity
plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Display the plot
"""