import torch
import torch.nn as nn


class FNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    criterion = nn.MSELoss()
    net.to(device)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs in trainloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, inputs)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader)}")


# berechnung von genauigkeit fehlt
def test(net, testloader, device: str):
    criterion = nn.MSELoss()
    net.eval()
    net.to(device)
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs in testloader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, inputs)  # Reconstruction loss
            total_loss += loss.item() * len(inputs)  # Add loss for current batch
            num_samples += len(inputs)  # Count total number of samples

    average_loss = total_loss / num_samples
    accuracy = 1.0 - average_loss  # Invert MSE to get a form of "accuracy"
    print("Accuracy: ", accuracy)
    return average_loss, accuracy


