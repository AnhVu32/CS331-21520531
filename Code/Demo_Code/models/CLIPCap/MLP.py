import torch.nn as nn

class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.Dropout(p=0.2))  # Update mới
        self.model = nn.Sequential(*layers)
