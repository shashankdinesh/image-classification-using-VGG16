from torch import optim
from model import *
from data_preparation import *


criteration = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Early stopping details
import numpy as np

n_epochs_stop = 5
min_val_loss = np.Inf
epochs_no_improve = 0
n_epochs = 200

# Main loop
for epoch in range(n_epochs):
    # Initialize validation loss for epoch
    val_loss = 0
    # Training loop
    for data, targets in dataloaders['train']:
        # Generate predictions
        out = model(data)
        # Calculate loss
        loss = criteration(out, targets)
        # Backpropagation
        loss.backward()
        # Update model parameters
        optimizer.step()

    # Validation loop
    for data, targets in dataloaders['val']:
        # Generate predictions
        out = model(data)
        # Calculate loss
        loss = criteration(out, targets)
        val_loss += loss

    # Average validation loss
    val_loss = val_loss / len(dataloaders['train'])
    print(val_loss)

    # If the validation loss is at a minimum
    if val_loss < min_val_loss:
        # Save the model
        torch.save(model, "/content/drive/My Drive/IMAGE_RECOGNITION/checkpoint_path")
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
        # Check early stopping condition
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            # Load in the best model
            # model = torch.load(checkpoint_path)

