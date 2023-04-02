"""latest version of SuperpointNet. Use it!

"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import copy
from models.unet_parts import *
from models.SA import ShuffleAttention
from models.ac import ACBlock
from models.pyconv import PyConv4

class SuperPointNet_ac_py(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, deploy=False):
        super(SuperPointNet_ac_py, self).__init__()
        c1, c2, c3, c4, d1 = 32, 64, 128, 256, 256
        det_h = 65

        self.deploy = deploy 
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)

        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1 = ACBlock(1, c1, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1_2 = ACBlock(c1, c1, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn1_2 = nn.BatchNorm2d(c1)

        self.conv2 = ACBlock(c1, c2, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv2_2 = ACBlock(c2, c2, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn2_2 = nn.BatchNorm2d(c2)

        self.conv3 = ACBlock(c2, c3, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv3_2 = ACBlock(c3, c3, kernel_size=3, padding=1, stride=1, deploy=self.deploy)
        self.bn3_2 = nn.BatchNorm2d(c3)

        self.conv5 = ShuffleAttention(c3, G=4)
        self.bn5 = nn.BatchNorm2d(c3)

        # Detector Head.
        self.convPa = PyConv4(c3, c4)
        self.bnPa = nn.BatchNorm2d(c4)
        self.convPb = torch.nn.Conv2d(c4, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = PyConv4(c3, c4)
        self.bnDa = nn.BatchNorm2d(c4)
        self.convDb = torch.nn.Conv2d(c4, d1, kernel_size=1, stride=1, padding=0)
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
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)

        x = self.relu(self.bn5(self.conv5(x)))

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # semi = self.pixel_shuffle(semi)
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc

def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

def main(weights_path, save_path):
  
  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  net = SuperPointNet_ac_py()
  checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
  net.load_state_dict(checkpoint['model_state_dict'])
  deploy_net = model_convert(net, save_path=save_path)
  print('==> Successfully deploy network.')
  return deploy_net

if __name__ == '__main__':
    main('ac_py_100.pth.tar','./weight/LEFPD_AC_PY_weight')
