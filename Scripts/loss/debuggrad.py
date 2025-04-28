import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import torch
import pytorch_lightning as pl
import kornia
#from tensordict.tensordict import TensorDict


class EMD(pl.LightningModule):
    """
    Compute the Earth Mover's Distance between two 1D discrete distributions (or observations).

    The EMD is a similarity metric between two probability distributions.
    In the discrete case, the Wasserstein distance can be understood as
    the cost of an optimal transport plan to convert one distribution into the other.
    The cost is calculated as the product of the amount of probability mass being moved and the distance it is being moved.

    When p=1, the Wasserstein distance is equivalent to the Earth Mover's Distance


    Args:
        X : 1d torch.Tensor
            A sample from a probability distribution or the support (set of all
            possible values) of a probability distribution. Each element is an
            observation or possible value.

        Y : 1d torch.Tensor
            A sample from or the support of a second distribution. Each element is an
            observation or possible value.

    Returns:
        distance : torch.tensor
            The computed distance between the distributions.
    """

    def __init__(self):
        super().__init__()


    def forward(self, X: torch.Tensor, Y: torch.Tensor):

        bins = torch.cat([X,Y], dim=0).unique().detach()

        hist_X = kornia.enhance.histogram(X[None,...], bins=bins,bandwidth=torch.tensor(0.9)).squeeze_(dim=0)
        hist_Y = kornia.enhance.histogram(Y[None,...], bins=bins,bandwidth=torch.tensor(0.9)).squeeze_(dim=0)

        PMF_X = hist_X/hist_X.sum()
        PMF_Y = hist_Y/hist_Y.sum()

        emd = torch.cumsum(PMF_X - PMF_Y, dim=-1).abs().sum()

        return emd

class TwoLayerSingleNeuron(nn.Module):
    def __init__(self):
        super(TwoLayerSingleNeuron, self).__init__()
        self.layer1 = nn.Linear(7, 7)
        self.layer2 = nn.Linear(7, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


X = torch.tensor([13.9989, 13.9233, 10.4435, 12.2223, 11.2133, 12.9421, 14.9323], requires_grad=True)
Y = torch.tensor([   14.0,    13.0,    11.0,    11.0,    10.0,    13.0,    15.0], requires_grad=True)


model = TwoLayerSingleNeuron()

criterion = EMD()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    print(outputs, loss)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Perda: {loss.item()}')

test_input = X
output = model(test_input)
print("Saída do neurônio após treinamento:", output)