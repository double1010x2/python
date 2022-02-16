from codecs import IncrementalDecoder
from functools import total_ordering
from operator import imatmul
from numpy.core.getlimits import iinfo
from numpy.lib.function_base import _update_dim_sizes
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt

import genQmat2 as gq

device = "cpu"
batch_size_list = [32, 64, 128]
batch_size = 256
# Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)

# Data
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=batch_size, shuffle=False)

n_mixer = 4
n_top = 0
n_qb = 6
cqmat = gq.genQubitMat(n_qb)
cqmat.genQMat(2)
n_qmat = cqmat.qmat.shape[0]
n_dim  = cqmat.qmat.shape[1]
ups = nn.Upsample(scale_factor = (n_dim*n_mixer/28.), mode="nearest")

qftmat = np.ones((n_dim, n_dim), np.complex)
for ri in range(qftmat.shape[0]):
    for ci in range(qftmat.shape[1]):
        qftmat[ri,ci] = (1j)**(ri*ci)/(n_dim)


q = 2
n_q = int(2**q)

# Model
class qNet(nn.Module):
    def __init__(self):
        super(qNet, self).__init__()
        dim_in    = n_qmat*n_q*n_mixer**2#784#n_qmat*n_q
        dim_l1    = int(dim_in*0.5)#128
        dim_l2    = int(dim_l1*0.5)#64
        self.main = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=dim_l1),
            nn.ReLU(),
            nn.Linear(in_features=dim_l1, out_features=dim_l2),
            nn.ReLU(),
            nn.Linear(in_features=dim_l2, out_features=10),
#            nn.Softmax(dim=1)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


qnet = qNet().to(device)
print(qnet)

# Parameters
epochs = 30
lr = 0.005
criterion = nn.NLLLoss()
optimizer = optim.SGD(qnet.parameters(), lr=lr, momentum=0.9)


imax   = 1
imin   = -1
igrid  = (imax-imin)/(n_q)

def splitByiGrid(imags):
    imags_new  = np.zeros((imags.shape[0], n_q, imags.shape[2], imags.shape[3]), np.float32)
    for qi in range(n_q):
        istart = imin+igrid*qi
        iend   = istart + igrid
        if qi >= (n_q-1):
            imags_new[:,qi,...] = np.logical_and(imags[:,0,...]>=istart, imags[:,0,...]<=iend).astype(np.float32)
        else:
            imags_new[:,qi,...] = np.logical_and(imags[:,0,...]>=istart, imags[:,0,...]<iend).astype(np.float32)
    return imags_new

gx, gy = np.mgrid[-8:8,-8:8]
gsigma = 4 
gmat   = np.exp(-(gx**2+gy**2)/(2*gsigma**2))

batch_num   = len(batch_size_list)
epoch_step  = epochs // batch_num + 1

def extractQfeature(data):
    n_batch     = len(data[0])
    data[0]     = ups(data[0]).numpy()
    data[0]     = splitByiGrid(data[0])
    _data       = np.empty((n_batch, n_q*n_mixer**2, n_dim, n_dim), np.float32)
    ni          = 0
    for rr in range(n_mixer):
        for cc in range(n_mixer):
            for channel in range(data[0].shape[1]):
                r0 = rr * n_dim 
                r1 = (rr+1) * n_dim 
                c0 = cc * n_dim 
                c1 = (cc+1) * n_dim 
                _data[:,ni,...] = data[0][:,channel,r0:r1,c0:c1]
                ni += 1
    data_sum    = _data.reshape(n_batch, _data.shape[1], -1).sum(axis=2)
    data_sum[data_sum<=0] = 1

    input       = np.zeros((n_batch, n_qmat*n_q*n_mixer**2), np.float32)
    ni = 0
    n_step = data_sum.shape[-1]
    for qi, qq in enumerate(cqmat.qmat):
        qfeature = _data[:,ni,...]*qq
        qfeature = qfeature.reshape(n_batch, -1)
        input[:,qi*n_step:(qi+1)*n_step]   = qfeature/data_sum
    input       = torch.from_numpy(input)
    input, labels  = input.to(device), data[1].to(device)
    return input, labels

for epoch in range(epochs):
    running_loss    = 0.
    total           = 0.
    correct         = 0.
    batch_size      = batch_size_list[epoch // epoch_step]
    trainSet = datasets.MNIST(root='MNIST', download=False, train=True, transform=transform)
    trainLoader = dset.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    for times, data in enumerate(trainLoader):
        inputs, labels  = extractQfeature(data) 
        optimizer.zero_grad()
        outputs     = qnet(inputs)
    
        loss = criterion(outputs, labels)
#        loss /= batch_size
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()/batch_size
        if times % 100 == 99 or times+1 == len(trainLoader):
            print(f'[{epoch+1}/{epochs}, {times+1}/{len(trainLoader)} loss: {loss.item():.3f}/{running_loss:.3f}')
        #    _, predicted = torch.max(outputs.data, 1)
        #    total += labels.size(0)
        #    correct += (predicted == labels).sum().item()
        #    print('Average Accuracy of the network on the training images: %d %%' % (100*correct / total))
        #    print(f'Accuracy of the network on the training images: {100*(predicted==labels).sum()/labels.size(0)} %%' )
print('Training Finished.')



# Test
correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = extractQfeature(data)

        outputs = qnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

with torch.no_grad():
    for data in testLoader:
        inputs, labels = extractQfeature(data)

        outputs = qnet(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
#            print(class_correct)
#            print(class_total)

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))