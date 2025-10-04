import torch

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluates the model's accuracy on a given dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the test data.
        device (str): The device to run the evaluation on ('cpu' or 'cuda').
    """
    # Send the model to the specified device
    model.to(device)
    # Set the model to evaluation mode (important for layers like dropout and batch norm)
    model.eval()

    correct = 0
    total = 0
    
    # Use torch.no_grad() to save memory and computation.
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # Make predictions
            outputs = model(images)
            
            # Get the class with the highest score as the prediction
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')