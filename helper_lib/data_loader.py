import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader(root_dir='./data', batch_size=64, train=True):
    """
    Creates a data loader for the CIFAR-10 dataset.

    Args:
        root_dir (str): The directory where the dataset will be stored.
        batch_size (int): The number of samples per batch.
        train (bool): If True, creates the training data loader. 
                      Otherwise, creates the test data loader.

    Returns:
        A torch.utils.data.DataLoader for the CIFAR-10 dataset.
    """
    # Converts images to PyTorch Tensors and normalizes the pixel values.
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download and load the appropriate CIFAR-10 dataset (train or test)
    dataset = torchvision.datasets.CIFAR10(root=root_dir, 
                                           train=train,
                                           download=True, 
                                           transform=transform)

    # Create the DataLoader, which groups the data into batches and shuffles it
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size,
                                              shuffle=train, # Only shuffle for the training set
                                              num_workers=2)

    return data_loader