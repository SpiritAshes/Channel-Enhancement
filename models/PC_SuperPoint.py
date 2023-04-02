"""latest version of SuperpointNet. Use it!

"""

import torch
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from models.unet_parts import *
from models.dcd import conv_basic_dy
from models.ea import External_attention
# from models.SubpixelNet import SubpixelNet
from models.pyconv import PyConv4, PyConv3_64_1, PyConv3_64_2, PyConv3_128
from  models.ECA import ECAAttention
class SuperPointNet_pc(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet_pc, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        # c1, c2, c3, c4, c5, d1 = 32, 64, 128, 128, 256, 256
        # det_h = 65
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv1 = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        # self.conv1 = torch.nn.Conv2d(1, c1, kernel_size=7, stride=1, padding=3)
        # self.bn1 = nn.BatchNorm2d(c1)
        # self.conv1_2 = torch.nn.Conv2d(c1, c1, kernel_size=5, stride=2, padding=2)
        # self.bn1_2 = nn.BatchNorm2d(c1)

        self.conv1 = PyConv3_64_1(1, c1, stride=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1_2 = PyConv3_64_2(c1, c1, stride=1)
        self.bn1_2 = nn.BatchNorm2d(c1)

        self.conv2 = PyConv3_64_2(c1, c2, stride=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv2_2 = PyConv3_64_2(c2, c2, stride=1)
        self.bn2_2 = nn.BatchNorm2d(c2)

        self.conv3 = PyConv3_128(c2, c3, stride=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv3_2 = PyConv3_128(c3, c3, stride=1)
        self.bn3_2 = nn.BatchNorm2d(c3)

        self.conv4 = PyConv3_128(c3, c4, stride=1)
        self.bn4 = nn.BatchNorm2d(c4)
        self.conv4_2 = PyConv3_128(c4, c4, stride=1)
        self.bn4_2 = nn.BatchNorm2d(c4)
        #

        # Detector Head.
        self.convPa = PyConv4(c4, c5, stride=1, pyconv_groups=[1, 1, 1, 1])
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, c5, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(c5)
        self.convPc = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPc = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = PyConv4(c4, c5, stride=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, c5, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(c5)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(9)
        self.convDc = PyConv4(c4, c5, stride=1)
        self.bnDc = nn.BatchNorm2d(c5)
        self.convDd = torch.nn.Conv2d(c5, c5, kernel_size=1, stride=1, padding=0)
        self.bnDd = nn.BatchNorm2d(c5)
        
        self.convDe = torch.nn.Conv2d(512, d1, kernel_size=1, stride=1, padding=0)
        self.bnDe = nn.BatchNorm2d(d1)

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
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x= self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))


        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        cPb = self.relu(self.bnPb(self.convPb(cPa)))
        semi = self.bnPc(self.convPc(cPb))

        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        cDb = self.relu(self.bnDb(self.convDb(cDa)))
        desc_H = x.shape[2]
        desc_W = x.shape[3]
        cDc = self.avgpool(x)
        cDc = self.relu(self.bnDc(self.convDc(cDc)))  
        cDc = self.relu(self.bnDd(self.convDd(cDc)))
        cDc = torch.nn.functional.interpolate(cDc, size = ([desc_H, desc_W]), mode = 'bilinear', align_corners = True)
        cDc = torch.cat((cDc, cDb), dim=1)    
        desc = self.bnDe(self.convDe(cDc))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc

 