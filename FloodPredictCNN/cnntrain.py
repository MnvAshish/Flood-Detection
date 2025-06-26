import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# Define transformations for the dataset with augmentation (optional)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load dataset (replace 'dataset_path' with your actual path)
dataset = datasets.ImageFolder(r"A:\fdataset\dataset", transform=transform)

# Calculate the split sizes (80% training, 10% validation, 10% testing)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split dataset into training, validation, and testing
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Loaders for train, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define your CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: (3, 128, 128) -> Output: (16, 128, 128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 128, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half -> (32, 64, 64)

        # Compute flatten size dynamically
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            dummy_output = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy_input)))))
            self.flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: 2 classes (Flooded, Non-Flooded)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Conv1
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + Pool -> Output: (32, 64, 64)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2
        return x


# Initialize model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'cnn_model.pth')  # Save model state_dict (recommended)

# Testing the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}")
print(f"Test Accuracy: {100 * correct / total:.2f}%")
