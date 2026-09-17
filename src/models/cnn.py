# cnn.py
#
# Minimal CNN regressor over the "grid" shower representation
# (data.features.grid.grid_generator(..., flatten=False) ->
# shape (N, 6, n, n): count / e_sum / e_mean / e_std / t_mean / t_std per
# spatial cell). Filled in as part of the architecture-comparison work
# (see search/run_search.py) -- was previously an empty stub, so
# ExperimentRunner.build_model() silently did nothing for model.type=="cnn".
#
# Kept intentionally simple (a handful of conv blocks + global average pool
# + linear head) so it trains fast enough to be one of several architectures
# in a hyperparameter search, not a state-of-the-art vision model.
import torch.nn as nn

from models.mlp import get_activation


def build_cnn_blocks(in_channels, n_blocks, base_channels, activation, dropout=0.0, batchnorm=False):
    layers = []
    channels = in_channels
    for i in range(max(n_blocks, 1)):
        out_channels = base_channels * (2 ** i)
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        channels = out_channels
    return layers, channels


class ShowerCNN(nn.Module):
    """CNN regressor for the grid shower representation.

    Parameters mirror ShowerMLP's naming so the same generic
    hyperparameter search (over "layers"/"neurons"/"activation") can drive
    either architecture: `n_blocks` plays the role of `n_layers`
    (network depth) and `base_channels` plays the role of `n_neurons`
    (network width) -- see ExperimentRunner.build_model().
    """

    def __init__(self, in_channels, output_dim, n_blocks=2, base_channels=16,
                 activation="relu", dropout=0.0, batchnorm=False):
        super().__init__()
        conv_layers, out_channels = build_cnn_blocks(
            in_channels, n_blocks, base_channels, activation, dropout, batchnorm
        )
        self.conv = nn.Sequential(*conv_layers)
        # Global average pooling makes the head independent of the grid
        # size `n`, so the same architecture works for grid_generator(n=8),
        # n=16, etc. without changes.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
