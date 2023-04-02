"""latest version of SuperpointNet. Use it!

"""
import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np
from models.da_att import PAM_Module
from models.da_att import CAM_Module
import torch

# from models.SubpixelNet import SubpixelNet
class SuperPointNet_DAN(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet_DAN, self).__init__()
        c1, c2, c3, c4, c5, c6, d1 = 32, 64, 128, 64, 128, 256, 256
        det_h = 65

        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        # Detector Head.
        self.conv1 = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.conv3 = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv4 = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(c2)
        self.conv5 = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c3)
        self.conv6 = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(c3)
        self.conv7 = torch.nn.Conv2d(c3, c4, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(c4)

        self.conv8 = torch.nn.Conv2d(c5, c5, 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(c5)
        self.conv9 = torch.nn.Conv2d(c5, c5, 3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(c5)

        self.sa = PAM_Module(c5)
        self.sc = CAM_Module(c5)
        self.conv10 = torch.nn.Conv2d(c5, c5, 3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(c5)
        self.conv11 = torch.nn.Conv2d(c5, c5, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(c5)

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv12 = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(c5)

        self.convPa = torch.nn.Conv2d(c5, c6, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c6)
        self.convPb = torch.nn.Conv2d(c6, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c5, c6, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c6)
        self.convDb = torch.nn.Conv2d(c6, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        R1 = x
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = F.interpolate(x, size=[R1.size(2), R1.size(3)], mode='bilinear', align_corners=True)
        x = R1 + x

        # feat1 = self.relu(self.bn8(self.conv8(x)))
        # sa_feat = self.sa(feat1)
        # sa_conv = self.relu(self.bn10(self.conv10(sa_feat)))
        #
        # feat2 = self.relu(self.bn9(self.conv9(x)))
        # sc_feat = self.sc(feat2)
        # sc_conv = self.relu(self.bn11(self.conv11(sc_feat)))
        #
        # feat_sum = sa_conv + sc_conv
        #
        #
        # x = self.relu(self.bn12(self.conv12(feat_sum)))

        x = self.relu(self.bn12(self.conv12(x)))
        feat1 = self.relu(self.bn8(self.conv8(x)))
        sa_feat = self.sa(feat1)
        sa_conv = self.relu(self.bn10(self.conv10(sa_feat)))

        feat2 = self.relu(self.bn9(self.conv9(x)))
        sc_feat = self.sc(feat2)
        sc_conv = self.relu(self.bn11(self.conv11(sc_feat)))

        feat_sum = sa_conv + sc_conv




        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(feat_sum)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(feat_sum)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc

