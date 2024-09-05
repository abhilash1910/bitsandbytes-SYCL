import torch
import torch.nn as nn
import torch.optim as optim
import python_src_quants 

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = python_src_quants.nn.Linear8bitLt(784, 256, has_fp16_weights=False)  # Using 8-bit Linear layer
        self.relu = nn.ReLU()
        self.fc2 = python_src_quants.nn.Linear8bitLt(256, 10, has_fp16_weights=False)  # Using 8-bit Linear layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN().xpu()
criterion = nn.CrossEntropyLoss()
optimizer = python_src_quants.optim.Adam8bit(model.parameters(), lr=0.001)  # Using 8-bit Adam optimizer

# Dummy dataset (e.g., MNIST)
# Replace this with a real dataset for actual training
train_loader = torch.utils.data.DataLoader(
    [(torch.randn(784), torch.tensor(0)) for _ in range(1000)], batch_size=32, shuffle=True
)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.xpu(), labels.xpu()

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        print("Loss")
        optimizer.step()

        running_loss += loss.item()
        print("Loss:", running_loss)
        
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Training complete.')

