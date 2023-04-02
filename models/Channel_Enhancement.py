"""latest version of SuperpointNet. Use it!

"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch

from models.unet_parts import *
from models.dcd import conv_basic_dy
from models.ea import External_attention
# from models.SubpixelNet import SubpixelNet
from models.ECA import ECAAttention
from models.rep import RepVGGBlock, repvgg_model_convert

class SuperPointNet_dcd(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, deploy=False):
        super(SuperPointNet_dcd, self).__init__()
        c1, c2, c3, c4, c5, d1 = 32, 64, 128, 256, 64, 256
        det_h = 65
        # c1, c2, c3, c4, c5, d1 = 32, 64, 128, 128, 256, 256
        # det_h = 65
        self.deploy = deploy
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)

        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1 = RepVGGBlock(1, c1, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)
        # self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = RepVGGBlock(c1, c1, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)
        # self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = RepVGGBlock(c1, c2, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)
        # self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = RepVGGBlock(c2, c2, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.conv5 = RepVGGBlock(c2, c3, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)

        self.conv6 = RepVGGBlock(c3, c3, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.conv7 = RepVGGBlock(c3, c4, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)

        self.pixel_shuffle = torch.nn.PixelShuffle(2)

        self.conv_attention = ECAAttention(3)

        self.convS_1 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnS_1 = nn.BatchNorm2d(det_h)
        self.convS_2 = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnS_2 = nn.BatchNorm2d(d1)

        # Detector Head.
        self.convDF_1 = torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        self.bnDF_1 = nn.BatchNorm2d(c5)
        self.convDF_2 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnDF_2 = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDD_1 = torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1)
        self.bnDD_1 = nn.BatchNorm2d(d1)
        self.convDD_2 = torch.nn.Conv2d(d1, d1, kernel_size=1, stride=1, padding=0)
        self.bnDD_2 = nn.BatchNorm2d(d1)


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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pixel_shuffle(x)
        x = self.pool(x)

        x_1 = self.bnS_1(self.convS_1(x))
        x_2 = self.bnS_2(self.convS_2(x))

        x = self.relu(self.conv_attention(x))

        # Detector Head.
        Head_DF = self.relu(self.bnDF_1(self.convDF_1(x)))
        # cPa = cPa + x_1  # 升维前融合
        semi = self.bnDF_2(self.convDF_2(Head_DF))
        semi = semi + x_1

        # Descriptor Head.
        Head_DD = self.relu(self.bnDD_1(self.convDD_1(x)))
        # cDa = cDa + x_2  # 升维前融合
        desc = self.bnDD_2(self.convDD_2(Head_DD))
        desc = desc + x_2
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return semi, desc




def main(weights_path, save_path):
  
  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  net = SuperPointNet_dcd()
  checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
  net.load_state_dict(checkpoint['model_state_dict'])
  deploy_net = repvgg_model_convert(net, save_path=save_path)
  print('==> Successfully deploy network.')
  return deploy_net

if __name__ == '__main__':
    main('dcd.pth.tar','./weight/dcd_weight')