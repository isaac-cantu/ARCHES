# mlp.py
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Activation {name} not supported")
    
def build_mlp_layers(input_dim, output_dim, n_layers, n_neurons, activation, dropout=0, batchnorm=False):
    
    layers = []

    # Primera capa
    layers.append(nn.Linear(input_dim, n_neurons))

    if batchnorm:
        layers.append(
            nn.BatchNorm1d(n_neurons)
        )

    layers.append(get_activation(activation))

    if dropout > 0:
        layers.append(nn.Dropout(0.5))

    # Capas ocultas
    for _ in range(n_layers):
        layers.append(nn.Linear(n_neurons, n_neurons))

        if batchnorm:
            layers.append(
                nn.BatchNorm1d(n_neurons)
            )

        layers.append(get_activation(activation))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Capa de salida
    layers.append(nn.Linear(n_neurons, output_dim))

    return layers

class ShowerMLP(nn.Module):

    def __init__(self, input_dim, output_dim, n_layers, n_neurons, activation, dropout=0, batchnorm=False):
        super().__init__()

        layers = build_mlp_layers(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)