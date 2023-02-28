# test 1)SageMaker Lab, 2)Colab and 3)local disk on read 5000 image from cifar10 trainingset
# cifar dataset is downloaded to the same "./data" fold, 

# Colab took 17.03 sec, (2nd fast download)
# SageMaker Lab took 16.69 sec,  (3rd fast download)
# Gradient notebook took 16.07 sec  (fastest download)
# Local MacBook took 2.36 sec (slowest download due to home network)

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#### time the pure data reading ####
from time import time
start = time()
for image in trainset:
    pass
end = time()
print("read {} image takes {} sec".format(len(trainset), end-start))
