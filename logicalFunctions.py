import matplotlib
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from CustomDataset import CustomDataset

torch.manual_seed(2)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)

xor_dataset = CustomDataset(X, Y)
xor_dataloader = DataLoader(xor_dataset)


class XOR(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.SiLU(),
            nn.Linear(2, output_dim)
        )
        self.weights_init()

    def forward(self, x):
        return self.linear_stack(x)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # initialize the weight tensor, here we use a normal distribution
                m.weight.data.normal_(0, 1)


model = XOR()
print(model)

loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()


epochs = 2001
steps = X.size(0)
for i in range(epochs):
    train(xor_dataloader, model, loss_func, optimizer)

for j in range(steps):
    x_var = Variable(X[j], requires_grad=False)
    y_var = Variable(Y[j], requires_grad=False)
    y_hat = model(x_var)
    loss = loss_func.forward(y_hat, y_var)
    print("Input is {}".format(x_var))
    print("Expected is {}".format(y_var[0]))
    print("Model says is {0:.2f}".format(y_hat[0]))
    print("Value is {0:.5f} off!\n".format(loss.item()))
