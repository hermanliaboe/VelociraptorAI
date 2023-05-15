import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.optim as optim
from torch_geometric.nn.conv import feast_conv
import numpy as np 


values_list = []  # create an empty list to store the values for each row
x = []
z = []
m = []
l = []

with open("data.txt", "r") as f:
    for line in f:  # loop over each line in the file
        values = line.strip().split("\t")  # split the line into a list of strings

        xL = []
        zL = []
        mL = []

        for i in range(20):
            xL.append(round(float(values[i]),3))
            zL.append(round(float(values[i+20]),3))
            mL.append(round(float(values[i+20*2]),3))
          
        x.append(xL)
        z.append(zL)
        m.append(mL)


dataY = []

dataX = []
for i in range(len(x)):
    dataXs = []
    for j in range(0,len(x[i])):
        dataX0 = []
        dataX0.append(j)
        dataX0.append(x[i][j])
        dataX0.append(z[i][j])
        dataXs.append(dataX0)
    dataX.append(dataXs)

dataX = np.array(dataX)
dataY = np.array(m)

dataEgdeIndex = []
for i in range(len(x[0])-1):
    dataEgdeIndex.append([i,i+1])
    dataEgdeIndex.append([i+1,i])  

dataEgdeIndex = np.array(dataEgdeIndex) 
dataEgdeIndex =np.transpose(dataEgdeIndex)


input_data = dataX
target_data = dataY
edge_index = dataEgdeIndex
dataX = torch.from_numpy(dataX)
dataY = torch.from_numpy(dataY)
edge_index = torch.from_numpy(edge_index)
edge_index = edge_index.to(torch.long)
dataX = dataX.to(torch.float)
dataY = dataY.to(torch.float)


dataset = []
for i in range(dataX.shape[0]):
    dataset.append(Data(x=dataX[i], edge_index=edge_index, y=dataY[i]))

train_loader = dataset[:200]
test_loader = dataset[200:]

class ArchNN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, heads, t_inv = True):
        super(ArchNN, self).__init__()
        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = feast_conv(16, 32, heads=heads, t_inv=t_inv)
        self.conv2 = feast_conv(32, 64, heads=heads, t_inv=t_inv)
        self.conv3 = feast_conv(64, 128, heads=heads, t_inv=t_inv)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #F.log_softmax(x, dijm=1)
        return x

import time
import torch
import torch.nn.functional as F


def print_info(info):
    message = ('Epoch: {}/{}, Duration: {:.3f}s,'
               'Train Loss: {:.4f}, Test Loss:{:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss'], info['test_loss'])
    print(message)


def run(model, train_loader, test_loader, num_nodes, epochs, optimizer):

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, train_loader, optimizer)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, num_nodes)
        eval_info = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration
        }

        print_info(eval_info)


def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        #log_probs = F.log_softmax(output, dim=1).double()
        #print(log_probs.dtype)
        #print(data.y.dtype)
        loss = F.mse_loss(output, data.y)
        #loss = F.nll_loss(log_probs, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, num_nodes):
    model.eval()
    correct = 0
    total_loss = 0
    n_graphs = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            out = model(data)
            total_loss += F.mse_loss(out, data.y).item()
            #pred = out.max(1)[1]
            #correct += pred.eq(data.y).sum().item()
            #n_graphs += data.num_graphs
    return total_loss / len(test_loader)

#runner
num_nodes = train_loader[0].x.shape[0]
num_features = train_loader[0].x.shape[1]

model = ArchNN(num_features, num_nodes, heads=10)

optimizer = optim.Adam(model.parameters(),
                       lr=0.001)


run(model, train_loader, test_loader, num_nodes, 10, optimizer)