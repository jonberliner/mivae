from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader

DIM_X = 784

### prep dataset and data transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(-1)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.view(-1)
    ])}

dataset = FashionMNIST('./FashionMNIST', download=True, transform=data_transforms['train'])
data_loader = DataLoader(dataset)
