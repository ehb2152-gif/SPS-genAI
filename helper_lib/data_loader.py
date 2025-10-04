import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader(dataset='cifar10', root_dir='./data', batch_size=64, train=True):
    if dataset.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_loader = torchvision.datasets.CIFAR10(root=root_dir, train=train,
                                                    download=True, transform=transform)

    elif dataset.lower() == 'mnist':
        transform = transforms.ToTensor()
        dataset_loader = torchvision.datasets.MNIST(root=root_dir, train=train,
                                                   download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported. Choose 'cifar10' or 'mnist'.")

    data_loader = torch.utils.data.DataLoader(dataset_loader, batch_size=batch_size,
                                              shuffle=train, num_workers=2)

    return data_loader