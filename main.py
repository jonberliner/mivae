import os
import sys
from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from torch.utils.data import Dataset, DataLoader
from torchvision.models import squeezenet1_1, resnet18
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy

INPUT_SIZE = 784
DIM_X = 3
SIZE_X = 28
BACKBONE_OUTPUT_SIZE = 256 * 2
Z1_SIZE = 13
Z2_SIZE = 17
DIM_Z = Z1_SIZE + Z2_SIZE


class Encoder(nn.Module):
    def __init__(self,
            backbone: nn.Module,
            dim_x: int,
            dim_z: int,
            backbone_output_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.backbone_output_dim = backbone_output_dim

        self.readout = nn.Linear(self.backbone_output_dim, self.dim_z)

    def forward(self, 
                inputs: torch.Tensor,
                return_backbone_outputs: Optional[bool]=False) -> torch.Tensor:
        bb_outputs = self.backbone(inputs)
        outputs = bb_outputs
        while len(outputs.shape) > 2:
            outputs = torch.mean(outputs, dim=-1)
        outputs = self.readout(outputs)
        if return_backbone_outputs:
            return maxes, bb_outputs
        else:
            return outputs

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x.expand(3, -1, -1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        lambda x: x.expand(3, -1, -1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset = FashionMNIST('./FashionMNIST', download=True, transform=data_transforms['train'])
dataloader = DataLoader(dataset)

squeezenet1 = squeezenet1_1(pretrained=True).features
squeezenet2 = squeezenet1_1(pretrained=True).features

# Z1_SIZE-way normal distr
encoder1 = Encoder(squeezenet1, INPUT_SIZE, Z1_SIZE * 2, BACKBONE_OUTPUT_SIZE)
# Z2_SIZE-way normal distr
encoder2 = Encoder(squeezenet2, INPUT_SIZE, Z2_SIZE * 2, BACKBONE_OUTPUT_SIZE)


inputs = dataset[0][0].unsqueeze(0)
inferred1 = encoder1(inputs)
inferred2 = encoder2(inputs)

p_z1_given_x = D.Normal(
        loc=inferred1[:, :Z1_SIZE], 
        scale=F.softplus(inferred1[:, Z1_SIZE:]) + 1e-4)

p_z2_given_x = D.Normal(
        loc=inferred2[:, :Z2_SIZE], 
        scale=F.softplus(inferred2[:, Z2_SIZE:]) + 1e-4)

z1_given_x = p_z1_given_x.rsample()
z2_given_x = p_z2_given_x.rsample()

z_given_x = torch.cat([z1_given_x, z2_given_x], dim=1)

class Decoder(nn.Module):
    def __init__(self,
            dim_z: int,
            dim_x: int,
            x_size: int=28) -> None:
        super().__init__()

        self.readin = nn.Conv2d(dim_z, 64, kernel_size=(1, 1), stride=(1, 1))
        self.resnet = resnet18(pretrained=False)
        self.readout = nn.Conv2d(512, dim_x, kernel_size=(1, 1), stride=(1, 1))
        self.model = nn.Sequential(*[
            self.readin,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.readout])

        self.dim_z = dim_z
        self.dim_x = dim_x
        self.x_size = x_size

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        inputs = torch.reshape(zs, (-1, self.dim_z, 1, 1))\
                      .expand(-1, self.dim_z, self.x_size*8, self.x_size*8)
        outputs = self.model(inputs)
        assert outputs.shape[2] == self.x_size
        return outputs
            
decoder = Decoder(DIM_Z, DIM_X, SIZE_X)
recon = decoder(z_given_x)
