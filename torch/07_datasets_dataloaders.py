import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

"""
When data is very large and can't fit into memory, we need to do batch wise processing
Few terms to know:
epoch = 1 forward and backward pass through all the training samples
batch_size = number of training examples in one forward and backward pass
num of iterations = number of passes, each pass using [batch_size] and number of samples
    - eg: 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
"""

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.: read with numpy or pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # here the first column is the class label, the rest are the features
        self.x = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # support indexing such that dataset[i] can be used to get i-th sample
        return self.x[index], self.y[index]
    
    def __len__(self):
        # we can call len(dataset) to return the size
        return self.n_samples
    
# create dataset
dataset = WineDataset()

# get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

# Load whole dataset with DataLoader
train_loader = DataLoader(dataset=dataset,
                          batch_size=4, 
                          shuffle=True, # shuffle: shuffle data, good for training
                          num_workers=0 # num_workers: faster loading with multiple subprocesses # !!! IF WE GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
                          )

# convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Run training process: forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}| inputs {inputs.shape} | labels {labels.shape}')
            
# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

# train_dataset = torchvision.datasets.MNIST(root='./data', 
#                                            train=True, 
#                                            transform=torchvision.transforms.ToTensor(),  
#                                            download=True)

# train_loader = DataLoader(dataset=train_dataset, 
#                                            batch_size=3, 
#                                            shuffle=True)