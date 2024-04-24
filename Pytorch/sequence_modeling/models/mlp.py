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
            output_dims,):
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


class MLPGenerator(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            hidden_units = [],
            hidden_activation = nn.ReLU()
    ):
        super(MLPGenerator, self).__init__()

        # model attributes
        self.model = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features

        # assemble model
        if hidden_units:
            # establish input layer
            self.model.append(nn.Linear(in_features, hidden_units[0]))
            self.model.append(hidden_activation)

            # set up remaining layers
            for idx in range(len(hidden_units)-1):
                in_size, out_size = hidden_units[idx], hidden_units[idx+1]
                self.model.append(nn.Linear(in_size, out_size))
                self.model.append(hidden_activation)
        else:
            # then model is a linear transformation
            hidden_units = [in_features]

        # output layer and final sequential
        output_units = out_features
        self.model.append(nn.Linear(hidden_units[-1], output_units))

    def forward(self, x):
        for f in self.model:
            x = f(x)
        return x




