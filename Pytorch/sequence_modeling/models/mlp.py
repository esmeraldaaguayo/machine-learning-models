import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRegression(nn.Module):
    def __init__(
            self,
            input_dims,
            hidden_dims1,
            hidden_dims2,
            hidden_dims3,
            output_dims=1,):
        super(MLPRegression, self).__init__()
        self.fc_linear1 = nn.Linear(input_dims, hidden_dims1)
        self.fc_linear2 = nn.Linear(hidden_dims1, hidden_dims2)
        self.fc_linear3 = nn.Linear(hidden_dims2, hidden_dims3)
        self.fc_linear4 = nn.Linear(hidden_dims3, output_dims)

    def forward(self, x):
        x = F.relu(self.fc_linear1(x))
        x = F.relu(self.fc_linear2(x))
        x = F.relu(self.fc_linear3(x))
        return self.fc_linear4(x)