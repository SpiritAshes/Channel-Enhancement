""" PyConv networks for image recognition as presented in our paper:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""
import torch
import torch.nn as nn


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3_64_1(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv3_64_1, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride)
        self.conv2_2 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride)
        self.conv2_3 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride)
        self.conv2_4 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class PyConv3_64_2(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv3_64_2, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class PyConv3_128(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv3_128, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = nn.Conv2d(inplans, planes//2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])


    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)