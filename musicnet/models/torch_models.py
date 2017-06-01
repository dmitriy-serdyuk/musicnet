from __future__ import print_function
import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam


class DeepConvnet(nn.Module):
    def __init__(self, window_size=4096, channels=1, output_size=84):
        super(DeepConvnet, self).__init__()
        self.conv_layers = []
        self.convs = []

        self.convs.append(nn.Conv1d(
                channels, 64, kernel_size=7, stride=3))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2, stride=2)])

        self.convs.append(nn.Conv1d(
                64, 64, kernel_size=3, stride=2))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            64, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            128, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            128, 256, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.ReLU(),
        ])

        self.convs.append(nn.Conv1d(
            256, 256, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.conv_sequential = nn.Sequential(*self.conv_layers)

        conv_shape = self._conv_shape((channels, window_size))
        self.linears = [nn.Linear(numpy.prod(conv_shape), 2048)]
        self.flat_layers = []
        self.flat_layers.extend([self.linears[-1], nn.ReLU()])
        self.linears.append(nn.Linear(2048, output_size))
        self.flat_layers.extend([self.linears[-1], nn.Sigmoid()])

        self.flat_sequential = nn.Sequential(*self.flat_layers)

        self.init()

    def summary(self):
        print("no summary yet")

    def init(self):
        for conv in self.convs:
            xavier_normal(conv.weight.data)
            constant(conv.bias.data, 0)

        for layer in self.linears:
            xavier_normal(layer.weight.data)
            constant(layer.bias.data, 0)

    def _forward_conv(self, input_):
        return self.conv_sequential(input_)

    def _conv_shape(self, input_shape):
        bs = 1
        input_ = Variable(torch.rand(bs, *input_shape))
        output = self._forward_conv(input_)
        return output.size()

    def forward(self, input_):
        conv_out = self._forward_conv(input_)
        return self.flat_sequential(conv_out.view(conv_out.size(0), -1))

    def cost(self, pred, target):
        return binary_cross_entropy(pred, target)


def train_model(iterator, model, steps_per_epoch, epochs, cuda=False,
                logger=None):
    optimizer = Adam(model.parameters(), lr=1.e-3)

    # in your training loop:
    for i, data in enumerate(iterator):
        input_, target = data

        input_ = Variable(torch.from_numpy(numpy.cast['float32'](input_.transpose((0, 2, 1)))))
        target = Variable(torch.from_numpy(numpy.cast['int64'](target)))
        if cuda:
            input_ = input_.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(input_)
        loss = model.cost(output, target)
        loss.backward()
        optimizer.step()

        if i >= steps_per_epoch * epochs:
            print('.. training finished')
            break

