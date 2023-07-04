import torch
import torch.nn as nn

torch.random.manual_seed(0)

A = nn.Linear(4,3)

x = torch.rand((2,4))
y = torch.rand((2, 3))

cf = nn.MSELoss()

opt = torch.optim.SGD(A.parameters(), lr=1e-1)
opt = torch.optim.Adam(A.parameters())
opt = torch.optim.LBFGS(A.parameters())

def closure():
    opt.zero_grad()
    loss = cf(A(x), y)
    loss.backward()
    return loss

for _ in range(20):
    l = opt.step(closure)
    print(l)

