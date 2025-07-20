import torch.nn as nn
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(131072, 128)  # to match the model checkpoint
 # Adjust according to your input size
        self.fc2 = nn.Linear(128, 2)           # Assuming binary classification
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    # 🔍 TEMP INPUT SIZE DEBUG
        dummy_input = torch.zeros(1, 3, 64, 64)  # Replace with actual image size
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        print("Flattened size after convs:", x.shape)  # Helps determine input size for fc1

        self.fc1 = nn.Linear(131072, 128)  # to match the model checkpoint
  # Update this based on above print
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
