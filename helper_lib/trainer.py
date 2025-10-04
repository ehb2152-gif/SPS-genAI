import torch

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Executes the training loop for a given model.

    Args:
        model (torch.nn.Module): The neural network model to train.
        data_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The optimization algorithm (e.g., optim.Adam).
        device (str): The device to run training on ('cpu' or 'cuda').
        epochs (int): The number of times to loop through the entire dataset.

    Returns:
        The trained model.
    """
    # Send the model to the specified device (GPU or CPU)
    model.to(device)
    # Set the model to training mode
    model.train()

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # Get the batch of inputs and labels, and move them to the device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Step 1: Zero the gradients for every batch
            optimizer.zero_grad()

            # Step 2: Make predictions (forward pass)
            outputs = model(inputs)

            # Step 3: Compute the loss
            loss = criterion(outputs, labels)

            # Step 4: Calculate the gradients (backward pass)
            loss.backward()

            # Step 5: Update the model's weights
            optimizer.step()

            # Print statistics to monitor training progress
            running_loss += loss.item()
            if (i + 1) % 200 == 0:  # Print every 200 mini-batches
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}], Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

    print('Finished Training')
    return model