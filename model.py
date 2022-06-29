import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Module, ModuleList, Conv1d, ConvTranspose1d


class ResidualBlock(torch.nn.Module):
    def __init__(self, dilation, channel=128, residual=True):
        super(ResidualBlock, self).__init__()
        self.residual = residual

        self.conv_gated_sigmoid = Conv1d(channel, channel, 3, padding='same', dilation=dilation)
        self.conv_gated_tanh = Conv1d(channel, channel, 3, padding='same', dilation=dilation)
        self.conv_skip_out = Conv1d(channel, channel, 1, padding='same')
        if self.residual:
            self.conv_residual = Conv1d(channel, channel, 1, padding='same')

    def forward(self, x):
        gated = torch.sigmoid(self.conv_gated_sigmoid(x)) * torch.tanh(self.conv_gated_tanh(x))
        skip_out = self.conv_skip_out(gated)
        if self.residual:
            res_out = self.conv_residual(gated) + x
        else:
            res_out = skip_out
        return res_out, skip_out

    def initialize_weights(self):
        init.xavier_uniform_(self.conv_gated_sigmoid.weight, init.calculate_gain('sigmoid'))
        init.xavier_uniform_(self.conv_gated_tanh.weight, init.calculate_gain('tanh'))
        init.xavier_uniform_(self.conv_residual.weight)
        init.kaiming_uniform_(self.conv_skip_out.weight, 0.1, 'fan_in', init.calculate_gain('leaky_relu', 0.1))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Wavenet(torch.nn.Module):
    def __init__(self, dilation, channel=128):
        super(Wavenet, self).__init__()

        self.conv_input = Conv1d(1, channel, 1, padding='same')
        residual_block_list = [ResidualBlock(d, channel) for d in dilation[:-1]]
        residual_block_list.append(ResidualBlock(dilation[-1], channel, False))
        self.residual_block = nn.ModuleList(residual_block_list)
        self.conv_output1 = Conv1d(channel, 2048, 3, padding='same')
        self.conv_output2 = Conv1d(2048, 256, 3, padding='same')
        self.conv_output3 = Conv1d(256, 1, 1, padding='same')

    def forward(self, x):
        x = self.conv_input(x)
        skip_out = None
        for f in self.residual_block:
            if skip_out is None:
                res_out, skip_out = f(x)
            else:
                res_out, s = f(res_out)
                skip_out += s
        skip_out = F.leaky_relu(skip_out, 0.1)
        output = F.leaky_relu(self.conv_output1(skip_out), 0.1)
        output = F.leaky_relu(self.conv_output2(output), 0.1)
        output = torch.tanh(self.conv_output3(output))
        return output

    def initialize_weights(self):
        init.xavier_uniform_(self.conv_input.weight)
        init.kaiming_uniform_(self.conv_output1.weight, 0.1, 'fan_in', init.calculate_gain('leaky_relu', 0.1))
        init.kaiming_uniform_(self.conv_output2.weight, 0.1, 'fan_in', init.calculate_gain('leaky_relu', 0.1))
        init.xavier_uniform_(self.conv_output3.weight, init.calculate_gain('tanh'))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    init.constant_(m.bias, 0)
