import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
   
def get_loaders_cifar(dataset_type="CIFAR10", img_width=224, img_height=224, batch_size=16):
 
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_width, img_height), scale=(0.05, 1.0)),  # Randomly crop the image
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
        transforms.RandomRotation(10),  # Random rotation within 10 degrees
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize between -1 and 1
    ])  # rescale train data between -1 to 1
 
    transform_test = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) # rescale test data between -1 to 1
    
    if dataset_type == "CIFAR10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
        
    elif dataset_type == "CIFAR100":
        trainset = datasets.CIFAR100(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)
        
    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    
    # No need to use `cycle()`, just return the DataLoader directly
    train_loader = DataLoader(trainset,
                    sampler=train_sampler,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True)
        
    test_loader = DataLoader(testset,
                    sampler=test_sampler,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True)
        
    return train_loader, test_loader, testset
