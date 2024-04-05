# transformer_test.py
import torch
import torch.nn as nn
from transformer import TransformerRegressor

# Hyperparameters
config ={
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'num_targets': 1,
    'batch_size': 1,
    'num_epochs': 10,
}

# Initialize the model
model = TransformerRegressor(config)

# Define a simple dataset
# For demonstration, we'll use random data
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        src = torch.randn(config['batch_size'], 10, config['d_model']) # Source sequence
        tgt = torch.randn(config['batch_size'], 10, config['d_model']) # Target sequence
        return src, tgt

# Create a DataLoader
dataset = SimpleDataset(100)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(config['num_epochs']):
    for src, tgt in dataloader:
        # Forward pass
        output = model(config, src, tgt)

        print(output.shape, tgt.shape)
        
        # Compute loss
        loss = criterion(output.squeeze(), tgt.squeeze())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {loss.item():.4f}")
