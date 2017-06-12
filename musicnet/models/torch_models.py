from __future__ import print_function
import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from torch.nn.functional import binary_cross_entropy, sigmoid, softplus
from torch.optim import Adam

from sklearn.metrics import average_precision_score, log_loss


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    def forward(self, input_, target):
        neg_abs = - input_.abs()
        loss = input_.clamp(min=0) - input_ * target + softplus(neg_abs)
        return loss.mean()


class DeepConvnet(nn.Module):
    def __init__(self, window_size=4096, channels=1, output_size=84):
        super(DeepConvnet, self).__init__()
        self.conv_layers = []
        self.convs = []

        self.convs.append(nn.Conv1d(
                channels, 64, kernel_size=7, stride=3))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([nn.BatchNorm1d(64, eps=1.e-3, momentum=1 - 0.99),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2, stride=2)])

        self.convs.append(nn.Conv1d(
                64, 64, kernel_size=3, stride=2))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(64, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            64, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            128, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128, eps=1.e-3, momentum=1 - 0.99),
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
            nn.BatchNorm1d(256, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.conv_sequential = nn.Sequential(*self.conv_layers)

        conv_shape = self._conv_shape((channels, window_size))
        self.linears = [nn.Linear(numpy.prod(conv_shape), 4096)]
        self.flat_layers = []
        self.flat_layers.extend([self.linears[-1], nn.ReLU()])
        self.linears.append(nn.Linear(4096, output_size))
        self.flat_layers.extend([nn.Dropout(0.8), self.linears[-1]])

        self.flat_sequential = nn.Sequential(*self.flat_layers)

        self.loss = StableBCELoss()

        self.init()

    def summary(self):
        print(self)

    def init(self):
        for conv in self.convs:
            xavier_normal(conv.weight.data)
            constant(conv.bias.data, 0)

        for layer in self.linears:
            xavier_normal(layer.weight.data)
            constant(layer.bias.data, 0)
        constant(layer.bias.data, -5.)

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
        return self.loss(pred.view(-1), target.view(-1))


class DeepConvnet2d(nn.Module):
    def __init__(self, window_size=4096, channels=1, output_size=84):
        super(DeepConvnet2d, self).__init__()
        self.conv_layers = []
        self.convs = []

        self.convs.append(nn.Conv1d(
                channels, 64, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([nn.BatchNorm1d(64, eps=1.e-3, momentum=1 - 0.99),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2, stride=2)])

        self.convs.append(nn.Conv1d(
                64, 64, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(64, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            64, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.convs.append(nn.Conv1d(
            128, 128, kernel_size=3, stride=1))
        self.conv_layers.append(self.convs[-1])
        self.conv_layers.extend([
            nn.BatchNorm1d(128, eps=1.e-3, momentum=1 - 0.99),
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
            nn.BatchNorm1d(256, eps=1.e-3, momentum=1 - 0.99),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        ])

        self.conv_sequential = nn.Sequential(*self.conv_layers)

        conv_shape = self._conv_shape((channels, window_size))
        self.linears = [nn.Linear(numpy.prod(conv_shape), 4096)]
        self.flat_layers = []
        self.flat_layers.extend([self.linears[-1], nn.ReLU()])
        self.linears.append(nn.Linear(4096, output_size))
        self.flat_layers.extend([nn.Dropout(0.8), self.linears[-1]])

        self.flat_sequential = nn.Sequential(*self.flat_layers)

        self.loss = StableBCELoss()

        self.init()

    def summary(self):
        print(self)

    def init(self):
        for conv in self.convs:
            xavier_normal(conv.weight.data)
            constant(conv.bias.data, 0)

        for layer in self.linears:
            xavier_normal(layer.weight.data)
            constant(layer.bias.data, 0)
        constant(layer.bias.data, -5.)

    def _forward_conv(self, input_):
        return self.conv_sequential(input_)

    def _conv_shape(self, input_shape):
        bs = 1
        input_ = Variable(torch.rand(bs, *input_shape))
        output = self._forward_conv(input_)
        return output.size()

    def forward(self, input_):
        # TODO: reshape input
        conv_out = self._forward_conv(input_)
        return self.flat_sequential(conv_out.view(conv_out.size(0), -1))

    def cost(self, pred, target):
        return self.loss(pred.view(-1), target.view(-1))


def validate(model, input_, output, logger, iteration, name):
    model.eval()
    preds = []
    input_size = input_.shape[0]
    N = 100
    input_chunk_size = input_size / N
    for i in range(N):
        inp = input_[(i * input_chunk_size): ((i + 1) * input_chunk_size)]
        pred = model(Variable(torch.from_numpy(inp)).cuda())
        preds.append(sigmoid(pred).data.cpu().numpy())
    pred = numpy.concatenate(preds, axis=0)
    try:
        average_precision = average_precision_score(
            output.flatten(), pred.flatten())
        loss = log_loss(output.flatten(), pred.flatten())
    except Exception:
        average_precision = 1.
        loss = 10.

    logger.log(
        {'iteration': iteration, 
         'records': {
             name: {'loss': loss,
                    'ap': average_precision}}})
    model.train()


def train_model(dataset, model, steps_per_epoch, epochs, cuda=False,
                logger=None, lr_schedule=None):
    optimizer = Adam(model.parameters(), lr=1.e-4)

    # in your training loop:
    iterator = dataset.train_iterator()
    Xvalid, Yvalid = dataset.eval_set('valid')
    Xvalid = Xvalid.transpose((0, 2, 1))
    Xvalid = numpy.ascontiguousarray(numpy.cast['float32'](Xvalid))
    Yvalid = numpy.ascontiguousarray(numpy.cast['float32'](Yvalid))

    Xtest, Ytest = dataset.eval_set('test')
    Xtest = Xtest.transpose((0, 2, 1))
    Xtest = numpy.ascontiguousarray(numpy.cast['float32'](Xtest))
    Ytest = numpy.ascontiguousarray(numpy.cast['float32'](Ytest))

    train_loss = 0.
    for i, data in enumerate(iterator):
        input_, target = data

        input_ = Variable(torch.from_numpy(numpy.cast['float32'](input_.transpose((0, 2, 1)))))
        target = Variable(torch.from_numpy(numpy.cast['float32'](target)))
        if cuda:
            input_ = input_.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(input_)
        loss = model.cost(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        if i % 100 == 0:
            logger.log(
                {'iteration': i, 
                 'records': {
                     'train': {'loss': train_loss / 100}}})
            train_loss = 0.

        if i % steps_per_epoch == 0:
            epoch = i / steps_per_epoch
            if lr_schedule is not None:
                lr = lr_schedule(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            print("\n\n Epoch {} finished\n\n".format(epoch))
            validate(model, Xvalid, Yvalid, logger, i, 'valid')
            validate(model, Xtest, Ytest, logger, i, 'test')
            print("\n\n Validation {} finished\n\n".format(epoch))

        if i >= steps_per_epoch * epochs:
            print('.. training finished')
            break

