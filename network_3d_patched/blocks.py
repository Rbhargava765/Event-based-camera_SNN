# This file is a patched copy of OF_EV_SNN-main/network_3d/blocks.py
# with imports modified to work without relative imports

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.activation_based import neuron

# Custom implementation of SEWResidual layer
class SEWResidual(nn.Module):
    def __init__(self, neurona, neuronb):
        """
        * :param neurona: spike neuron for activation function
        * :param neuronb: spike neuron for residual function

        """
        super().__init__()
        self.neurona = neurona
        self.neuronb = neuronb

    def forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor):
        """
        :param x_seq: input to the activation function
        :param y_seq: input to the residual function
        :return: output of SEW-ResNet: neurona(x) + neuronb(y)

        """
        return self.neurona(x_seq) + self.neuronb(y_seq)


###############
### SCALING ###
###############

class MultiplyBy(nn.Module):

    def __init__(self, scale_value: float = 5., learnable: bool = False) -> None:
        super(MultiplyBy, self).__init__()

        if learnable:
            self.scale_value = Parameter(Tensor([scale_value]))
        else:
            self.scale_value = scale_value

    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.scale_value)




################
### CONV 2-D ###
################

class SeparableBlock3x3(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, leak_mem=True):
        super(SeparableBlock3x3, self).__init__()

        self.dephtwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=3 // 2,
            stride=stride,
            bias=False,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.leak_mem = leak_mem

        if leak_mem:
            self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)
        else:
            self.lif1 = neuron.IFNode(v_threshold=0.5, v_reset=0.0)

        self.multiplier = MultiplyBy(5.)

    def forward(self, x):
        x = self.dephtwise(x)
        x = self.pointwise(x)
        x = self.multiplier(x)
        x = self.lif1(x)
        return x


class SeparableBlock5x5(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, leak_mem=True):
        super(SeparableBlock5x5, self).__init__()

        self.dephtwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=5 // 2,
            stride=stride,
            bias=False,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.leak_mem = leak_mem

        if leak_mem:
            self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)
        else:
            self.lif1 = neuron.IFNode(v_threshold=0.5, v_reset=0.0)

        self.multiplier = MultiplyBy(5.)

    def forward(self, x):
        x = self.dephtwise(x)
        x = self.pointwise(x)
        x = self.multiplier(x)
        x = self.lif1(x)
        return x


class SeparableBlock7x7(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, leak_mem=True):
        super(SeparableBlock7x7, self).__init__()

        self.dephtwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=7 // 2,
            stride=stride,
            bias=False,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.leak_mem = leak_mem

        if leak_mem:
            self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=0.5, v_reset=0.0)
        else:
            self.lif1 = neuron.IFNode(v_threshold=0.5, v_reset=0.0)

        self.multiplier = MultiplyBy(5.)

    def forward(self, x):
        x = self.dephtwise(x)
        x = self.pointwise(x)
        x = self.multiplier(x)
        x = self.lif1(x)
        return x




################
### RESIDUAL ###
################

class SeparableSEWResBlock(nn.Module):

    def __init__(self, in_channels: int, multiply_factor: float = 5., kernel_size: int = 3):
        super(SeparableSEWResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            MultiplyBy(multiply_factor)
        )

        self.sew1 = SEWResidual(
            neuron.IFNode(v_threshold=float('inf'), v_reset=0.),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0.)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
                groups=in_channels
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            MultiplyBy(multiply_factor)
        )

        self.sew2 = SEWResidual(
            neuron.IFNode(v_threshold=1., v_reset=0.),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0.)
        )

    def forward(self, x):
        out_sew1 = self.sew1(self.conv1(x), x)
        return self.sew2(self.conv2(out_sew1), out_sew1)


if __name__ == '__main__':
    x = torch.randn((32, 512, 32, 32))
    net = SeparableSEWResBlock(512)
    with torch.no_grad():
        out = net(x) 